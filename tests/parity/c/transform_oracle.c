/* libheif transform oracle of av1-rs (CI only, docs/parity.md "Transforms"). Written from the
 * public libheif API (LGPL-3.0 library, linked statically into this CI tool, never shipped); no
 * libheif code is copied. The shape follows the transform oracle of isobmff-rs.
 *
 *   heif-transform-oracle <in-dir> <out-dir>
 *
 * Decodes the primary image of every file of <in-dir> (flat) with the transforms applied by
 * libheif and writes, for <stem> = the name without its extension, the layout of the oracle of
 * av1-rs (examples/oracle.rs):
 *   <out-dir>/<stem>.yuv     planes Y, Cb, Cr one after the other, 8-bit samples as bytes, deeper
 *                            samples as little-endian u16; monochrome writes Y only
 *   <out-dir>/<stem>.frames  one line: W H bitdepth layout (400, 420, 422, 444)
 *   <out-dir>/<stem>.err     instead of both, when libheif cannot decode the file
 * Colourspace and chroma are left undefined and the nclx passthrough is on, so libheif keeps the
 * native planes; the alpha plane is not written. An image above 64 Mpixel is not decoded.
 */
#define _POSIX_C_SOURCE 200809L
#include <dirent.h>
#include <libheif/heif.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PIXELS ((uint64_t)64 << 20)

static const char *g_out;

static void put(const char *stem, const char *ext, const char *text)
{
    char p[4096];
    FILE *f;
    snprintf(p, sizeof p, "%s/%s.%s", g_out, stem, ext);
    f = fopen(p, "wb");
    if (f) {
        fputs(text, f);
        fclose(f);
    }
}

static const char *write_image(const heif_image *img, const char *stem)
{
    heif_colorspace cs = heif_image_get_colorspace(img);
    heif_chroma ch = heif_image_get_chroma_format(img);
    heif_channel chans[3] = {heif_channel_Y, heif_channel_Cb, heif_channel_Cr};
    const char *layout;
    int n, i, bd, bps;
    char p[4096], line[128];
    FILE *f;

    if (cs == heif_colorspace_monochrome && ch == heif_chroma_monochrome) {
        layout = "400";
        n = 1;
    } else if (cs == heif_colorspace_YCbCr && ch == heif_chroma_420) {
        layout = "420";
        n = 3;
    } else if (cs == heif_colorspace_YCbCr && ch == heif_chroma_422) {
        layout = "422";
        n = 3;
    } else if (cs == heif_colorspace_YCbCr && ch == heif_chroma_444) {
        layout = "444";
        n = 3;
    } else {
        return "format";
    }
    bd = heif_image_get_bits_per_pixel_range(img, heif_channel_Y);
    bps = bd > 8 ? 2 : 1;
    if ((uint64_t)heif_image_get_width(img, heif_channel_Y) * (uint64_t)heif_image_get_height(img, heif_channel_Y) > MAX_PIXELS) {
        return "toolarge";
    }
    snprintf(p, sizeof p, "%s/%s.yuv", g_out, stem);
    f = fopen(p, "wb");
    if (!f) {
        return "io";
    }
    for (i = 0; i < n; i++) {
        int w = heif_image_get_width(img, chans[i]), h = heif_image_get_height(img, chans[i]), y;
        size_t stride = 0, x;
        const uint8_t *base = heif_image_get_plane_readonly2(img, chans[i], &stride);
        if (!base || w <= 0 || h <= 0 || stride < (size_t)w * (size_t)bps) {
            fclose(f);
            return "plane";
        }
        for (y = 0; y < h; y++) {
            const uint8_t *row = base + (size_t)y * stride;
            if (bps == 1) {
                fwrite(row, 1, (size_t)w, f);
            } else {
                for (x = 0; x < (size_t)w; x++) { /* little-endian, whatever the host is */
                    uint16_t v;
                    uint8_t b[2];
                    memcpy(&v, row + 2 * x, 2);
                    b[0] = (uint8_t)(v & 0xff);
                    b[1] = (uint8_t)(v >> 8);
                    fwrite(b, 1, 2, f);
                }
            }
        }
    }
    if (fclose(f) != 0) {
        return "io";
    }
    snprintf(line, sizeof line, "%d %d %d %s\n", heif_image_get_width(img, heif_channel_Y),
             heif_image_get_height(img, heif_channel_Y), bd, layout);
    put(stem, "frames", line);
    return NULL;
}

static void do_file(const char *dir, const char *name)
{
    char path[4096], stem[4096];
    heif_context *ctx;
    heif_image_handle *h = NULL;
    heif_image *img = NULL;
    heif_decoding_options *opts;
    const char *tok = "decode";
    char *dot;

    snprintf(path, sizeof path, "%s/%s", dir, name);
    snprintf(stem, sizeof stem, "%s", name);
    dot = strrchr(stem, '.');
    if (dot) {
        *dot = 0;
    }
    ctx = heif_context_alloc();
    if (!ctx) {
        return;
    }
    heif_context_set_security_limits(ctx, heif_get_disabled_security_limits());
    opts = heif_decoding_options_alloc();
    opts->output_image_nclx_profile_passthrough = 1;
    if (heif_context_read_from_file(ctx, path, NULL).code == heif_error_Ok &&
        heif_context_get_primary_image_handle(ctx, &h).code == heif_error_Ok) {
        int iw = heif_image_handle_get_ispe_width(h), ih = heif_image_handle_get_ispe_height(h);
        if (iw < 0 || ih < 0 || (uint64_t)iw * (uint64_t)ih > MAX_PIXELS) {
            tok = "toolarge";
        } else if (heif_decode_image(h, &img, heif_colorspace_undefined, heif_chroma_undefined, opts).code == heif_error_Ok) {
            tok = write_image(img, stem);
            heif_image_release(img);
        }
        heif_image_handle_release(h);
    }
    if (tok) {
        put(stem, "err", tok);
    }
    heif_decoding_options_free(opts);
    heif_context_free(ctx);
}

int main(int argc, char **argv)
{
    DIR *d;
    struct dirent *e;
    if (argc != 3) {
        fprintf(stderr, "usage: heif-transform-oracle <in-dir> <out-dir>\n");
        return 2;
    }
    g_out = argv[2];
    d = opendir(argv[1]);
    if (!d) {
        fprintf(stderr, "heif-transform-oracle: %s\n", argv[1]);
        return 2;
    }
    heif_init(NULL);
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] != '.') {
            do_file(argv[1], e->d_name);
        }
    }
    closedir(d);
    heif_deinit();
    return 0;
}
