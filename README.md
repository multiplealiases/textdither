# textdither

"Dither" (more properly posterize) binary data using k-means clustering.

This thing is probably not getting published on
PyPI, so install it straight from Git.

```console
$ pipx install .
```

## Usage

(`tranz` is a text file containing the lyrics to
*Tranz* by Gorillaz. It is not included in this repo.)

```console
$ textdither tranz --token-length 5 --codebook-size 80 --samples 10000 --threshold 0.0001
Oscillate toujkgke phememuedho you're io youuedh
Dprfhememe rij dancljlfOscilne thtoujk your edge
When you get back on a Shtsrdjv lkget bne the rijm is hememm is
Do yjv lkpg jkke me?
Do you dack ogke pd
Do wpr jwpr jgke pwpr jhemem?
Do you dance ljlfthis?
Dprfis?
FSee yhtsrdke uhme
(...)
```
