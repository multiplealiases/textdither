import argparse
import numpy
import sklearn.cluster
import more_itertools
import sys


def main():
    rng = numpy.random.default_rng()

    parser = argparse.ArgumentParser(
        prog="text2dither", description="Dither text using k-means, fitting against codebook to construct reference"
    )
    parser.add_argument("reference", type=argparse.FileType('rb'))
    parser.add_argument("codebook", type=argparse.FileType('rb'))
    parser.add_argument("-t", "--token-length", default=5, type=int)
    parser.add_argument("-c", "--codebook-size", default=10, type=int)
    # limit the number of samples taken to improve processing speed
    parser.add_argument("-n", "--samples", default=50000, type=int)
    parser.add_argument("-i", "--iterations", default=100, type=int)
    args = parser.parse_args()

    reference_tokens = numpy.array(list((make_tokens(args.reference, args.token_length))))
    tokens = numpy.array(list((make_tokens(args.codebook, args.token_length))))

    tokenlen = len(tokens)
    samples = min(args.samples, tokenlen)

    alltokens = tokens.astype(float)
    traintokens = rng.choice(alltokens, size=samples)

    reftokens = reference_tokens.astype(float)

    codebook = sklearn.cluster.MiniBatchKMeans(n_clusters=args.codebook_size, max_iter=args.iterations).fit(traintokens)
    # print(codebookint, file=sys.stderr)
    indexes = codebook.predict(reftokens)
    # print(indexes[0], file=sys.stderr)
    for i in indexes:
        out = codebook.cluster_centers_[i].astype(numpy.uint8)
        sys.stdout.buffer.write(out)
    # deeply hacky way to pad out the output for FFmpeg's use
    nulls = bytes([0] * args.token_length)
    sys.stdout.buffer.write(nulls)


def make_tokens(file, token_length):
    filebytes = bytes(file.read())
    tokens = more_itertools.grouper(filebytes, token_length, incomplete="ignore")
    return tokens
