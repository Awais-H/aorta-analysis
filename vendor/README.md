# Vendored wheels

`setup.sh` installs from this directory with `--no-index` when it contains
wheels, which is what makes the install work on a machine with no internet
access. This path is tested: a fresh virtualenv installed entirely from here
runs the pipeline with no network.

The directory is empty in version control on purpose. The wheel set is
specific to one platform and one Python version and runs to roughly 145 MB per
combination, so committing a guess is both large and likely to be the wrong
guess.

Populate it for the evaluation environment before submitting:

    tools/vendor_wheels.sh --python-version 3.11     # 64-bit Linux, the default

`vendor/*.whl` is in `.gitignore` so that a local, wrong-platform download does
not get committed by accident. When you have the right wheels, add them
explicitly:

    git add -f vendor/*.whl

Then verify with a network-disabled run of `./setup.sh`.
