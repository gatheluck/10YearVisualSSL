# Pinned downstream encoder environment; container build not yet verified.
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
COPY methods/vjepa2_1/requirements.lock.txt /opt/lock/method.txt
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
# exists and misreports what is in it.
COPY bin/verify-environment.py /opt/verify-environment.py
RUN python3 /opt/verify-environment.py \
        --lock /opt/lock/method.txt \
        --lock /opt/lock/tools.txt

WORKDIR /work
COPY . /work

ENV PYTHONPATH=/work
