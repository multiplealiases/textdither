import argparse
import numpy
import scipy
import more_itertools
import sys


def main():
    rng = numpy.random.default_rng()

    parser = argparse.ArgumentParser(
        prog="textdither", description="Dither text using k-means"
    )
    parser.add_argument("file", type=argparse.FileType('rb'))
    parser.add_argument("-t", "--token-length", default=5, type=int)
    parser.add_argument("-c", "--codebook-size", default=10, type=int)
    # limit the number of samples taken to improve processing speed
    parser.add_argument("-n", "--samples", default=5000, type=int)
    parser.add_argument("-p", "--threshold", default=1e-03, type=float)
    args = parser.parse_args()

    tokens = numpy.array(list((make_tokens(args.file, args.token_length))))

    tokenlen = len(tokens)
    samples = min(args.samples, tokenlen)

    alltokens = tokens.astype(float)
    traintokens = rng.choice(alltokens, size=samples)
    codebook = scipy.cluster.vq.kmeans(traintokens, args.codebook_size, thresh=args.threshold)
    codebookint = codebook[0].astype(numpy.uint8)
    # print(codebookint, file=sys.stderr)
    indexes = scipy.cluster.vq.vq(alltokens, codebook[0])
    # print(indexes[0], file=sys.stderr)
    for i in indexes[0]:
        out = codebookint[i]
        sys.stdout.buffer.write(out)
    # deeply hacky way to pad out the output for FFmpeg's use
    nulls = bytes([0] * args.token_length)
    sys.stdout.buffer.write(nulls)


def make_tokens(file, token_length):
    filebytes = bytes(file.read())
    tokens = more_itertools.grouper(filebytes, token_length, incomplete="ignore")
    return tokens
