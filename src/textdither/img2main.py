import argparse
import numpy
import itertools
import sklearn.cluster
import skimage.util
import sys


def main():
    rng = numpy.random.default_rng()

    parser = argparse.ArgumentParser(
        prog="textdither", description="Dither text using k-means"
    )
    parser.add_argument("reference", type=argparse.FileType('rb'))
    parser.add_argument("codebook", type=argparse.FileType('rb'))
    parser.add_argument("-t", "--token-size", required=True)
    parser.add_argument("-b", "--bytes-per-pixel", required=True, type=int)
    parser.add_argument("-r", "--reference-size", required=True)
    parser.add_argument("-C", "--codebookimg-size", required=True)
    parser.add_argument("-c", "--codebook-size", default=10, type=int)
    # limit the number of samples taken to improve processing speed
    parser.add_argument("-n", "--samples", default=50000, type=int)
    parser.add_argument("-i", "--iterations", default=5, type=int)
    args = parser.parse_args()

    width, height = parse_res(args.reference_size)
    codebook_width, codebook_height = parse_res(args.codebookimg_size)
    token_width, token_height = parse_res(args.token_size)

    # output is (block_count_w, block_count_h, bpp, block_width, block_height, bpp)
    print("making tokens", file=sys.stderr)
    tmp = make_tokens(args.reference, height, width, token_height, token_width, args.bytes_per_pixel)
    block_count = tmp.shape[0] * tmp.shape[1]
    block_dim = args.bytes_per_pixel * token_height * token_width

    codebooktmp = make_tokens(args.codebook, codebook_height, codebook_width, token_height, token_width, args.bytes_per_pixel)
    codebook_block_count = codebooktmp.shape[0] * codebooktmp.shape[1]
    codebook_block_dim = args.bytes_per_pixel * token_height * token_width

    print("reshaping tokens", file=sys.stderr)
    tokens = tmp.reshape(block_count, block_dim)
    codebook_tokens = codebooktmp.reshape(codebook_block_count, codebook_block_dim)

    tokenlen = len(codebook_tokens)
    samples = min(args.samples, tokenlen)

    alltokens = codebook_tokens.astype(float)
    traintokens = rng.choice(alltokens, size=samples)

    print("clustering tokens", file=sys.stderr)
    codebook = sklearn.cluster.MiniBatchKMeans(n_clusters=args.codebook_size, max_iter=args.iterations).fit(traintokens)
    indexes = codebook.predict(tokens)
    print(codebook.cluster_centers_, file=sys.stderr)

    framebuffer = numpy.zeros((height, width, args.bytes_per_pixel))
    print(framebuffer.shape, file=sys.stderr)

    print("splatting tokens to framebuffer", file=sys.stderr)
    x = range(0, width, token_width)
    y = range(0, height, token_height)
    offsets = list(itertools.product(y, x))
    print(len(offsets), file=sys.stderr)
    for i in indexes:
        entry = codebook.cluster_centers_[i].reshape(tmp.shape[3], tmp.shape[4], args.bytes_per_pixel)
        paste(framebuffer, entry, (offsets[pos][0], offsets[pos][1]))
    sys.stdout.buffer.write(framebuffer.astype(numpy.uint8))


def make_tokens(file, img_height, img_width, block_width, block_height, bpp):
    image = numpy.frombuffer(file.read(), dtype=numpy.uint8).reshape(img_height, img_width, bpp)
    tiles = skimage.util.view_as_blocks(image, block_shape=(block_width, block_height, bpp))
    return tiles


# https://stackoverflow.com/a/50692782
def paste_slices(tup):
    pos, w, max_w = tup
    wall_min = max(pos, 0)
    wall_max = min(pos + w, max_w)
    block_min = -min(pos, 0)
    block_max = max_w - max(pos + w, max_w)
    block_max = block_max if block_max != 0 else None
    return slice(wall_min, wall_max), slice(block_min, block_max)


def paste(wall, block, loc):
    loc_zip = zip(loc, block.shape, wall.shape)
    wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
    wall[wall_slices] = block[block_slices]
# ^---end of copied section---^


def parse_res(res):
    ret = res.split("x")
    assert len(ret) == 2, "{} is not a valid resolution"
    return list(map(int, ret))
