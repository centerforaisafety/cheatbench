#! /usr/bin/env bash
# Runs inside the container's mount namespace just before switchroot, with
# ENROOT_ROOTFS pointing at the per-episode overlay. Everything it writes lands
# in that overlay's tmpfs upper layer and dies with the episode.
#
# This replaces enroot's stock hooks, all of which are dropped: 10-shadow.sh in
# particular copies the HOST user's passwd entry into the container, so
# `cat /etc/passwd` would have handed the agent the operator's username and
# home directory (<operator>:x:...:/data/<operator>:/bin/bash).
set -eu

# The conventional /dev/ptmx. Our /dev is a fresh tmpfs, so the image's own
# symlink is hidden and a bind of the host's would import the host devpts
# instance (ptmxmode=000, i.e. unusable).
ln -sfn pts/ptmx "${ENROOT_ROOTFS}/dev/ptmx"

# Resolvers only. The host's resolv.conf also carries `search`/`domain` lines
# naming the cluster's internal domain; those say more about where this is
# running than an ordinary container would know. When the mirror is up
# core/sandbox/runtime.py overwrites this file with slirp4netns' forwarder
# anyway.
if [ -f /etc/resolv.conf ]; then
    grep -E '^[[:space:]]*nameserver[[:space:]]' /etc/resolv.conf \
        > "${ENROOT_ROOTFS}/etc/resolv.conf" || : > "${ENROOT_ROOTFS}/etc/resolv.conf"
fi

# A stock container /etc/hosts. Never the host's: that one lists the node and
# its neighbours by name.
if [ ! -s "${ENROOT_ROOTFS}/etc/hosts" ]; then
    printf '127.0.0.1\tlocalhost\n::1\tlocalhost ip6-localhost ip6-loopback\n' \
        > "${ENROOT_ROOTFS}/etc/hosts"
fi
rm -f "${ENROOT_ROOTFS}/etc/hostname"
