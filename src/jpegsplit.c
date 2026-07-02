/*
 * jpegsplit.c
 *
 * libjpeg-turbo Modifications:
 * Copyright (C) 2026, libjpeg-turbo contributors.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains a command-line tool that splits a JPEG image into a
 * grid of fixed-size tiles using lossless DCT-domain cropping.  Optionally,
 * tile encoding can be parallelized across multiple workers using OpenMP.
 *
 * Usage:
 *   jpegsplit [switches] inputfile
 *
 * Switches:
 *   -tile WxH       Tile size in pixels (default: 2048x2048)
 *   -outdir DIR     Output directory (default: .)
 *   -workers N      Number of parallel workers (0 = all cores, default: 0)
 *   -maxmemory N    Maximum memory for DCT arrays in kbytes (default: unlimited)
 *   -optimize       Optimize Huffman table (smaller file, but slower)
 *   -arithmetic     Use arithmetic coding
 *   -version        Print version information and exit
 *
 * Output files are named <outdir>/<basename>_<row>_<col>.jpg, where
 * <basename> is the input filename without its directory prefix or extension.
 *
 * Tile offsets are always at multiples of the tile dimensions, so tile
 * dimensions must be multiples of the image's iMCU size (computed
 * automatically from the source file's chroma subsampling).  If the
 * user-specified dimensions are not compatible, they are rounded up
 * automatically and a warning is printed.
 *
 * Edge tiles are smaller than the nominal tile size when the image
 * dimensions are not exact multiples of the tile size; they carry the
 * exact remaining pixel count (rounded to the DCT block boundary by the
 * codec, as with any normal JPEG).
 *
 * MEMORY MODEL
 * ============
 * The source JPEG is decoded ONCE into DCT coefficient virtual arrays
 * (libjpeg's internal format).  For a 65000x65000 4:2:0 image this is
 * roughly 12 GB; libjpeg will spill to temporary files when the arrays
 * exceed -maxmemory.
 *
 * Each worker thread requires only a single tile-sized workspace (~tile_w x
 * tile_h DCT blocks, a few MB), NOT another full-image decode.  So total
 * peak memory is approximately:
 *
 *   full-image DCT  +  n_workers * tile_workspace
 *
 * For example, a 65k x 65k image with 2048x2048 tiles and 8 workers needs
 * roughly 12 GB + 8 * 12 MB ≈ 12.1 GB, versus 8 * 12 GB = 96 GB with the
 * naive per-tile decode approach.
 *
 * NOTE: parallel execution of jtransform_execute_transform is safe only when
 * the DCT virtual arrays are fully resident in RAM (no disk spill).  If you
 * use -maxmemory to cap memory and the image is very large, use -workers 1
 * to avoid concurrent seeks on the spill file.
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include "cdjpeg.h"             /* Common decls for cjpeg/djpeg applications */
#include "transupp.h"           /* Support routines for jpegtran */
#include "jversion.h"           /* for version message */
#include "jconfigint.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <string.h>


#define DEFAULT_TILE_WIDTH   2048
#define DEFAULT_TILE_HEIGHT  2048
/* Minimum user-specified tile dimension (must be a multiple of the DCT
 * block size).  The actual minimum enforced at runtime is the iMCU size. */
#define MIN_TILE_SIZE        8


static const char *progname;    /* program name for error messages */


LOCAL(void)
usage(void)
{
  fprintf(stderr, "usage: %s [switches] inputfile\n", progname);
  fprintf(stderr, "Switches (names may be abbreviated):\n");
  fprintf(stderr, "  -tile WxH       Tile size in pixels (default: %dx%d)\n",
          DEFAULT_TILE_WIDTH, DEFAULT_TILE_HEIGHT);
  fprintf(stderr,
          "                  Width and height must be multiples of %d\n",
          MIN_TILE_SIZE);
  fprintf(stderr,
          "                  (tool adjusts upward to match the iMCU size)\n");
  fprintf(stderr, "  -outdir DIR     Output directory (default: .)\n");
  fprintf(stderr, "  -copy none      Copy no extra markers from source file (default)\n");
  fprintf(stderr, "  -copy comments  Copy only comment (COM) markers\n");
  fprintf(stderr, "  -copy icc       Copy only ICC profile (APP2) markers\n");
  fprintf(stderr, "  -copy all       Copy all extra markers\n");
  fprintf(stderr,
          "                  EXIF ImageWidth/Height patched per tile;\n");
  fprintf(stderr,
          "                  other EXIF tags still reference the full\n");
  fprintf(stderr,
          "                  source image (GPS, camera model, etc.)\n");
  fprintf(stderr,
          "  -workers N      Parallel workers (0 = all cores, default: 0)\n");
  fprintf(stderr,
          "  -maxmemory N    Max RAM for DCT arrays in kbytes (default: unlimited)\n");
  fprintf(stderr,
          "                  When exceeded, libjpeg spills to temp files.\n");
  fprintf(stderr,
          "                  Use -workers 1 when spilling to avoid file-seek races.\n");
#ifdef ENTROPY_OPT_SUPPORTED
  fprintf(stderr,
          "  -optimize       Optimize Huffman table (smaller file, but slower compression)\n");
#endif
#ifdef C_ARITH_CODING_SUPPORTED
  fprintf(stderr, "  -arithmetic     Use arithmetic coding\n");
#endif
  fprintf(stderr, "  -version        Print version information and exit\n");
  exit(EXIT_FAILURE);
}


/* Extract the filename base (no directory prefix, no extension) from path.
 * Result is written to out[] and NUL-terminated, truncated to outlen-1 chars.
 */
LOCAL(void)
get_basename(const char *path, char *out, size_t outlen)
{
  const char *start = path;
  const char *p;
  const char *dot;
  size_t len;

  for (p = path; *p != '\0'; p++) {
    if (*p == '/' || *p == '\\')
      start = p + 1;
  }
  dot = strrchr(start, '.');
  len = dot ? (size_t)(dot - start) : strlen(start);
  if (len >= outlen)
    len = outlen - 1;
  memcpy(out, start, len);
  out[len] = '\0';
}


/* Encode one tile using a pre-decoded source.
 *
 * srcinfo        : fully decoded decompressor (jpeg_read_coefficients done).
 * src_coef_arrays: the coefficient arrays from jpeg_read_coefficients.
 * xform          : crop transform, already configured for this tile
 *                  (output_width/height and x/y_crop_offset set by caller).
 * outpath        : output file path.
 *
 * Thread-safety: safe to call concurrently as long as
 *   (a) each thread has a different xform->workspace_coef_arrays, and
 *   (b) the DCT virtual arrays are fully in RAM (no disk spill).
 *
 * For -copy all: jtransform_adjust_parameters would normally patch
 * ExifImageWidth/Height in-place inside srcinfo->marker_list->data, which is
 * shared across threads.  We avoid the race by giving each tile a shallow
 * copy of srcinfo whose marker_list points to a private copy of the EXIF
 * bytes.  Only those bytes are duplicated; everything else (virtual arrays,
 * memory manager, all other markers) is shared read-only and is safe.
 *
 * Returns 0 on success, non-zero on failure.
 */
LOCAL(int)
encode_tile(j_decompress_ptr srcinfo, jvirt_barray_ptr *src_coef_arrays,
            jpeg_transform_info *xform, const char *outpath,
            boolean optimize_coding, boolean arith_code,
            JCOPY_OPTION copyoption)
{
  struct jpeg_compress_struct dstinfo;
  struct jpeg_error_mgr jdsterr;
  jvirt_barray_ptr *dst_coef_arrays;
  FILE *fp;
  /* Private EXIF copy for -copy all (see comment above). */
  j_decompress_ptr effective_src = srcinfo;
  struct jpeg_decompress_struct tile_src;
  struct jpeg_marker_struct tile_marker;
  JOCTET *tile_exif = NULL;

  /* When copying all markers, jtransform_adjust_parameters patches the
   * ExifImageWidth/Height TIFF tags inside srcinfo->marker_list->data.
   * That buffer is shared by all threads.  To avoid a data race without
   * any serialization overhead, we give this tile a shallow copy of
   * srcinfo whose first marker node points to a private EXIF data buffer.
   * All other markers (XMP, IPTC, ...) are only read, so sharing them
   * is safe.  The expensive DCT copy + Huffman encode still runs fully
   * in parallel because it uses the original srcinfo's virtual arrays. */
  if ((copyoption == JCOPYOPT_ALL || copyoption == JCOPYOPT_ALL_EXCEPT_ICC) &&
      srcinfo->marker_list != NULL &&
      srcinfo->marker_list->marker == JPEG_APP0 + 1 &&
      srcinfo->marker_list->data_length >= 6 &&
      srcinfo->marker_list->data[0] == 0x45 &&  /* 'E' */
      srcinfo->marker_list->data[1] == 0x78 &&  /* 'x' */
      srcinfo->marker_list->data[2] == 0x69 &&  /* 'i' */
      srcinfo->marker_list->data[3] == 0x66 &&  /* 'f' */
      srcinfo->marker_list->data[4] == 0 &&
      srcinfo->marker_list->data[5] == 0) {
    tile_exif = (JOCTET *)malloc(srcinfo->marker_list->data_length);
    if (tile_exif != NULL) {
      memcpy(tile_exif, srcinfo->marker_list->data,
             srcinfo->marker_list->data_length);
      tile_marker        = *srcinfo->marker_list; /* copy the list node */
      tile_marker.data   = tile_exif;             /* redirect to private buf */
      tile_src           = *srcinfo;              /* shallow-copy decompressor */
      tile_src.marker_list = &tile_marker;        /* override first marker */
      effective_src      = &tile_src;
    }
  }

  dstinfo.err = jpeg_std_error(&jdsterr);
  jpeg_create_compress(&dstinfo);

  jpeg_copy_critical_parameters(srcinfo, &dstinfo);

#ifdef ENTROPY_OPT_SUPPORTED
  dstinfo.optimize_coding = optimize_coding;
#else
  (void)optimize_coding;
#endif
#ifdef C_ARITH_CODING_SUPPORTED
  dstinfo.arith_code = arith_code;
#else
  (void)arith_code;
#endif

  /* Adjust dstinfo dimensions and get the destination coefficient pointer.
   * Uses effective_src so the EXIF patch goes into the private buffer. */
  dst_coef_arrays = jtransform_adjust_parameters(effective_src, &dstinfo,
                                                 src_coef_arrays, xform);

  if ((fp = fopen(outpath, WRITE_BINARY)) == NULL) {
    fprintf(stderr, "%s: can't open %s for writing\n", progname, outpath);
    jpeg_destroy_compress(&dstinfo);
    free(tile_exif);
    return 1;
  }

  /* Write the tile (same order as jpegtran) */
  jpeg_stdio_dest(&dstinfo, fp);
  jpeg_write_coefficients(&dstinfo, dst_coef_arrays);
  jcopy_markers_execute(effective_src, &dstinfo, copyoption);
  /* DCT copy + Huffman encode: uses original srcinfo for virtual array access */
  jtransform_execute_transform(srcinfo, &dstinfo, src_coef_arrays, xform);
  jpeg_finish_compress(&dstinfo);
  jpeg_destroy_compress(&dstinfo);

  fclose(fp);
  free(tile_exif);  /* NULL-safe */

  return jdsterr.num_warnings ? 1 : 0;
}


int
main(int argc, char **argv)
{
  int argn;
  char *arg;
  const char *infilename = NULL;
  const char *outdir = ".";
  JDIMENSION tile_w = DEFAULT_TILE_WIDTH;
  JDIMENSION tile_h = DEFAULT_TILE_HEIGHT;
  int n_workers = 0;
  long max_memory = 0;          /* 0 = unlimited */
  boolean optimize_coding = FALSE;
  boolean arith_code = FALSE;
  JCOPY_OPTION copyoption = JCOPYOPT_NONE;  /* -copy switch */
  FILE *fp;
  struct jpeg_decompress_struct srcinfo;
  struct jpeg_error_mgr jsrcerr;
  jvirt_barray_ptr *src_coef_arrays;
  JDIMENSION img_w, img_h;
  JDIMENSION imcu_w, imcu_h;
  int n_cols, n_rows, n_tiles, tile_idx;
  int n_actual_workers, i, errors;
  char basename_buf[256];
  jpeg_component_info *ci;
  jpeg_transform_info *xforms;   /* per-worker xform (workspace pointer) */
  struct cdjpeg_progress_mgr src_progress;

  progname = argv[0];
  if (progname == NULL || progname[0] == '\0')
    progname = "jpegsplit";

  /* Parse command-line switches */
  for (argn = 1; argn < argc; argn++) {
    arg = argv[argn];
    if (*arg != '-') {
      /* Not a switch: must be the input file name */
      if (infilename != NULL) {
        fprintf(stderr, "%s: only one input file allowed\n", progname);
        usage();
      }
      infilename = arg;
      continue;
    }
    arg++;  /* skip the '-' */

    if (keymatch(arg, "arithmetic", 1)) {
      /* Use arithmetic coding */
#ifdef C_ARITH_CODING_SUPPORTED
      arith_code = TRUE;
#else
      fprintf(stderr, "%s: sorry, arithmetic coding not supported\n",
              progname);
      exit(EXIT_FAILURE);
#endif

    } else if (keymatch(arg, "optimize", 1) ||
               keymatch(arg, "optimise", 1)) {
      /* Optimize Huffman tables */
#ifdef ENTROPY_OPT_SUPPORTED
      optimize_coding = TRUE;
#else
      fprintf(stderr,
              "%s: sorry, entropy optimization was not compiled\n", progname);
      exit(EXIT_FAILURE);
#endif

    } else if (keymatch(arg, "copy", 2)) {
      /* Select which extra markers to copy to each tile */
      if (++argn >= argc)
        usage();
      if (keymatch(argv[argn], "none", 1))
        copyoption = JCOPYOPT_NONE;
      else if (keymatch(argv[argn], "comments", 1))
        copyoption = JCOPYOPT_COMMENTS;
      else if (keymatch(argv[argn], "icc", 1))
        copyoption = JCOPYOPT_ICC;
      else if (keymatch(argv[argn], "all", 1))
        copyoption = JCOPYOPT_ALL;
      else
        usage();

    } else if (keymatch(arg, "outdir", 4)) {
      /* Set output directory */
      if (++argn >= argc)
        usage();
      outdir = argv[argn];

    } else if (keymatch(arg, "maxmemory", 3)) {
      /* Maximum memory for DCT virtual arrays (kbytes). */
      char ch = 'x';
      if (++argn >= argc)
        usage();
      if (sscanf(argv[argn], "%ld%c", &max_memory, &ch) < 1 ||
          max_memory < 0) {
        fprintf(stderr, "%s: bogus -maxmemory argument '%s'\n",
                progname, argv[argn]);
        usage();
      }
      if (ch == 'm' || ch == 'M')
        max_memory *= 1000L;

    } else if (keymatch(arg, "tile", 4)) {
      /* Tile dimensions: WxH or WXH */
      unsigned int tw = 0, th = 0;
      char sep = '\0';
      if (++argn >= argc)
        usage();
      if (sscanf(argv[argn], "%u%c%u", &tw, &sep, &th) != 3 ||
          (sep != 'x' && sep != 'X') || tw == 0 || th == 0) {
        fprintf(stderr, "%s: bogus -tile argument '%s'\n",
                progname, argv[argn]);
        usage();
      }
      if (tw % MIN_TILE_SIZE != 0 || th % MIN_TILE_SIZE != 0) {
        fprintf(stderr,
                "%s: tile dimensions must be multiples of %d\n",
                progname, MIN_TILE_SIZE);
        exit(EXIT_FAILURE);
      }
      tile_w = (JDIMENSION)tw;
      tile_h = (JDIMENSION)th;

    } else if (keymatch(arg, "workers", 1)) {
      /* Number of parallel worker threads */
      if (++argn >= argc)
        usage();
      if (sscanf(argv[argn], "%d", &n_workers) != 1 || n_workers < 0) {
        fprintf(stderr, "%s: bogus -workers argument '%s'\n",
                progname, argv[argn]);
        usage();
      }
#ifndef _OPENMP
      if (n_workers > 1)
        fprintf(stderr,
                "%s: warning: built without OpenMP;"
                " -workers N > 1 has no effect\n", progname);
#endif

    } else if (keymatch(arg, "version", 4)) {
      fprintf(stderr, "%s version %s (build %s)\n",
              PACKAGE_NAME, VERSION, BUILD);
      exit(EXIT_SUCCESS);

    } else {
      fprintf(stderr, "%s: unknown switch '-%s'\n", progname, arg);
      usage();
    }
  }

  if (infilename == NULL) {
    fprintf(stderr, "%s: must specify an input file\n", progname);
    usage();
  }

  /* ---- Open source file and create the (single) decompressor ---- */

  if ((fp = fopen(infilename, READ_BINARY)) == NULL) {
    fprintf(stderr, "%s: can't open %s for reading\n", progname, infilename);
    exit(EXIT_FAILURE);
  }

  srcinfo.err = jpeg_std_error(&jsrcerr);
  jpeg_create_decompress(&srcinfo);

  if (max_memory > 0)
    srcinfo.mem->max_memory_to_use = max_memory * 1000L;

  jpeg_stdio_src(&srcinfo, fp);
  jcopy_markers_setup(&srcinfo, copyoption);
  (void)jpeg_read_header(&srcinfo, TRUE);

  img_w = srcinfo.image_width;
  img_h = srcinfo.image_height;

  /* Compute the iMCU dimensions from the maximum sampling factors */
  {
    int max_h = 1, max_v = 1;
    for (i = 0; i < srcinfo.num_components; i++) {
      ci = &srcinfo.comp_info[i];
      if (ci->h_samp_factor > max_h) max_h = ci->h_samp_factor;
      if (ci->v_samp_factor > max_v) max_v = ci->v_samp_factor;
    }
    imcu_w = (JDIMENSION)(max_h * DCTSIZE);
    imcu_h = (JDIMENSION)(max_v * DCTSIZE);
  }

  if (img_w == 0 || img_h == 0) {
    fprintf(stderr, "%s: invalid image dimensions in %s\n",
            progname, infilename);
    jpeg_destroy_decompress(&srcinfo);
    fclose(fp);
    exit(EXIT_FAILURE);
  }

  /* ---- Validate and adjust tile dimensions ---- */

  /* Round tile dimensions up to the next multiple of the iMCU size so that
   * all tile offsets land on iMCU boundaries (required for lossless crop). */
  if (tile_w % imcu_w != 0) {
    JDIMENSION new_w = ((tile_w + imcu_w - 1) / imcu_w) * imcu_w;
    fprintf(stderr,
            "%s: warning: tile width %u adjusted to %u"
            " (iMCU width = %u)\n",
            progname, tile_w, new_w, imcu_w);
    tile_w = new_w;
  }
  if (tile_h % imcu_h != 0) {
    JDIMENSION new_h = ((tile_h + imcu_h - 1) / imcu_h) * imcu_h;
    fprintf(stderr,
            "%s: warning: tile height %u adjusted to %u"
            " (iMCU height = %u)\n",
            progname, tile_h, new_h, imcu_h);
    tile_h = new_h;
  }

  /* If the tile is at least as large as the image, produce a single tile.
   * Offset is 0, which is always iMCU-aligned. */
  if (tile_w > img_w) tile_w = img_w;
  if (tile_h > img_h) tile_h = img_h;
  if (tile_w == 0) tile_w = 1;
  if (tile_h == 0) tile_h = 1;

  /* ---- Compute the tile grid ---- */

  n_cols  = (int)((img_w + tile_w - 1) / tile_w);
  n_rows  = (int)((img_h + tile_h - 1) / tile_h);
  n_tiles = n_cols * n_rows;

  get_basename(infilename, basename_buf, sizeof(basename_buf));

  fprintf(stderr,
          "%s: %ux%u image -> %d tile(s) in a %d-col x %d-row grid"
          " (tile size %ux%u, iMCU %ux%u)\n",
          progname, img_w, img_h, n_tiles, n_cols, n_rows,
          tile_w, tile_h, imcu_w, imcu_h);

  /* ---- Determine actual worker count ---- */

  n_actual_workers = 1;
#ifdef _OPENMP
  if (n_workers > 0) {
    omp_set_num_threads(n_workers);
    n_actual_workers = n_workers;
  } else {
    n_actual_workers = omp_get_max_threads();
  }
#endif
  if (n_actual_workers > n_tiles)
    n_actual_workers = n_tiles;

  /* ---- Pre-allocate one workspace per worker ----
   *
   * All workspaces live in srcinfo's JPOOL_IMAGE (freed at
   * jpeg_finish_decompress).  Each workspace holds ONE tile's worth of DCT
   * blocks: O(tile_w * tile_h * sizeof(JCOEF)).  This is done BEFORE
   * jpeg_read_coefficients so the virtual-array manager can plan the layout.
   *
   * For the tile at pixel offset (0, 0), jtransform_execute_transform is a
   * no-op and the compressor reads directly from src_coef_arrays, so no
   * workspace is needed.  We null out that slot in encode_tile.
   *
   * If the image fits in a single tile there are no non-(0,0) tiles and no
   * workspace is needed at all; we allocate a dummy slot with NULL.
   */

  xforms = (jpeg_transform_info *)
    malloc((size_t)n_actual_workers * sizeof(jpeg_transform_info));
  if (xforms == NULL) {
    fprintf(stderr, "%s: out of memory allocating xform array\n", progname);
    jpeg_destroy_decompress(&srcinfo);
    fclose(fp);
    exit(EXIT_FAILURE);
  }

  for (i = 0; i < n_actual_workers; i++) {
    /* Proto crop: use a non-zero offset to force workspace allocation.
     * We use (tile_w, 0) when there are multiple columns, else (0, tile_h)
     * when there are multiple rows, else (0, 0) for a single tile. */
    JDIMENSION proto_x = (n_cols > 1) ? tile_w : 0;
    JDIMENSION proto_y = (proto_x == 0 && n_rows > 1) ? tile_h : 0;

    memset(&xforms[i], 0, sizeof(xforms[i]));
    xforms[i].transform        = JXFORM_NONE;
    xforms[i].crop             = TRUE;
    xforms[i].crop_width       = tile_w;
    xforms[i].crop_width_set   = JCROP_POS;
    xforms[i].crop_height      = tile_h;
    xforms[i].crop_height_set  = JCROP_POS;
    xforms[i].crop_xoffset     = proto_x;
    xforms[i].crop_xoffset_set = JCROP_POS;
    xforms[i].crop_yoffset     = proto_y;
    xforms[i].crop_yoffset_set = JCROP_POS;

    if (!jtransform_request_workspace(&srcinfo, &xforms[i])) {
      fprintf(stderr, "%s: failed to allocate workspace for worker %d\n",
              progname, i);
      free(xforms);
      jpeg_destroy_decompress(&srcinfo);
      fclose(fp);
      exit(EXIT_FAILURE);
    }
    /* xforms[i].workspace_coef_arrays is now set (or NULL for single tile) */
  }

  /* ---- Decode source DCT coefficients ONCE ----
   *
   * libjpeg reads the entire image into virtual arrays.  If the arrays
   * exceed srcinfo.mem->max_memory_to_use, libjpeg spills to temp files.
   * After this call the source file is fully consumed and can be closed.
   */
  fprintf(stderr, "%s: decoding %s ...\n", progname, infilename);
  start_progress_monitor((j_common_ptr)&srcinfo, &src_progress);
  src_progress.report = TRUE;
  src_coef_arrays = jpeg_read_coefficients(&srcinfo);
  end_progress_monitor((j_common_ptr)&srcinfo);
  fclose(fp);  /* source fully in virtual arrays; file no longer needed */

  /* ---- Process all tiles, in parallel when safe ---- */

  fprintf(stderr, "%s: encoding %d tile(s) with %d worker(s) ...\n",
          progname, n_tiles, n_actual_workers);

  errors = 0;

#pragma omp parallel for schedule(dynamic) reduction(+:errors)
  for (tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
    int tid = 0;
    int row, col;
    JDIMENSION x, y, w, h;
    jpeg_transform_info cur_xform;
    char outpath[4096];

#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    row = tile_idx / n_cols;
    col = tile_idx % n_cols;
    x   = (JDIMENSION)((unsigned int)col * tile_w);
    y   = (JDIMENSION)((unsigned int)row * tile_h);
    w   = (x + tile_w <= img_w) ? tile_w : (img_w - x);
    h   = (y + tile_h <= img_h) ? tile_h : (img_h - y);

    /* Build a per-invocation copy of the xform so threads don't share
     * mutable fields (output_width/height, crop offsets).  The
     * workspace_coef_arrays pointer, which is the only per-worker resource,
     * comes from the pre-allocated slot for this thread. */
    cur_xform = xforms[0];   /* copy base fields (transform, crop flags, ...) */
    cur_xform.crop_width    = w;
    cur_xform.crop_height   = h;
    cur_xform.crop_xoffset  = x;
    cur_xform.crop_yoffset  = y;
    cur_xform.output_width  = w;   /* valid: x is always iMCU-aligned */
    cur_xform.output_height = h;
    cur_xform.x_crop_offset = x / imcu_w;  /* in iMCU units */
    cur_xform.y_crop_offset = y / imcu_h;

    /* For the (0,0) tile jtransform_execute_transform is a no-op and
     * jtransform_adjust_parameters must return src_coef_arrays so the
     * compressor reads directly from the source.  Null the workspace so
     * jtransform_adjust_parameters takes the "return src" path. */
    if (x == 0 && y == 0)
      cur_xform.workspace_coef_arrays = NULL;
    else
      cur_xform.workspace_coef_arrays = xforms[tid].workspace_coef_arrays;

    snprintf(outpath, sizeof(outpath), "%s/%s_%d_%d.jpg",
             outdir, basename_buf, row, col);

    if (encode_tile(&srcinfo, src_coef_arrays, &cur_xform, outpath,
                    optimize_coding, arith_code, copyoption) != 0) {
      errors++;
    } else {
      fprintf(stderr, "%s: wrote %s (%ux%u)\n", progname, outpath, w, h);
    }
  }

  /* ---- Clean up ---- */

  free(xforms);

  /* jpeg_finish_decompress releases the JPOOL_IMAGE (workspaces + DCT arrays) */
  (void)jpeg_finish_decompress(&srcinfo);
  jpeg_destroy_decompress(&srcinfo);

  exit(errors ? EXIT_FAILURE : EXIT_SUCCESS);
  return 0;                     /* suppress no-return-value warnings */
}
