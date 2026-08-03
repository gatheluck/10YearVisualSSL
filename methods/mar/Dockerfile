# The locked environment for mar, as an image.
#
# Build from the repository root, because the tooling lock, bin/ and the pinned
# upstream under third_party/ all live there:
#
#   git submodule update --init third_party/mar   # populate the pinned model
#   docker build -f methods/mar/Dockerfile -t mar .
#
# **The submodule must be checked out before building.** `COPY . /work` copies
# the working tree, and third_party/mar is empty until the submodule is
# initialised; the adapter imports the upstream model from there at run time.
#
# Podman takes the same command. For Apptainer, convert the built image or pull
# it from a registry; see the method's README.
#
# **This definition has never been built.** There is no Docker, Podman or
# Apptainer on the machine it was written on. Everything about it is checked by
# reading -- tests/test_container.py -- and nothing by running.

# Pinned by digest, not by tag. It matches .python-version at the repository
# root, which is what the cp312 wheels in the locks require.
FROM python:3.12.13-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

# No apt at all. The wheels in the lock are self-contained, and any
# `apt-get install` without pinned versions would make two builds differ.

# A fresh virtual environment, not the base image's site-packages.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The locks are copied before the source so that editing code does not
# reinstall torch.
COPY methods/mar/requirements.lock.txt /opt/lock/method.txt
COPY requirements-tools.lock.txt /opt/lock/tools.txt

# The same pair the README installs. Naming only one of them is the mistake
# that once made a correct environment look broken.
RUN pip install --no-cache-dir --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r /opt/lock/method.txt \
        -r /opt/lock/tools.txt

# **The build proves its own environment.** If what got installed is not what
# the locks describe, the build fails here rather than producing an image that
# exists and misreports what is in it. The pinned upstream is imported through
# PYTHONPATH, never pip-installed, so it is not in the lock and does not trip
# this check.
COPY bin/verify-environment.py /opt/verify-environment.py
RUN python3 /opt/verify-environment.py \
        --lock /opt/lock/method.txt \
        --lock /opt/lock/tools.txt

WORKDIR /work
COPY . /work

# The adapter runs from the method directory and reaches the repository root
# (adapterlib) and the pinned upstream (third_party/mar) through PYTHONPATH.
ENV PYTHONPATH=/work

# No ENTRYPOINT. Apptainer and Docker disagree about how one is invoked, and a
# plain image can be driven the same way by both:
#
#   docker run --rm -v "$PWD/runs:/runs" -w /work/methods/mar \
#       mar python3 -m adapter --config /runs/resolved.json --out /runs/out
#
#   apptainer exec --bind "$PWD/runs:/runs" mar.sif \
#       sh -c 'cd /work/methods/mar && python3 -m adapter \
#              --config /runs/resolved.json --out /runs/out'
CMD ["python3", "-c", "import torch; print('torch', torch.__version__)"]
