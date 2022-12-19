#!/usr/bin/env bash

# Run this script to set up local Git hooks for this project.

egrep-q() {
	egrep "$@" >/dev/null 2>/dev/null
}

die() {
	echo 1>&2 "$@" ; exit 1
}

# Make sure we are inside the repository.
cd "${BASH_SOURCE%/*}" &&

# Populate ".git/hooks".
echo 'Setting up git hooks...' &&
git_dir=$(git rev-parse --git-dir) &&
mkdir -p "$git_dir/hooks" &&
cd "$git_dir/hooks" &&
cp -ap "../../tools/hooks/." . &&
if ! test -e .git; then
	git init -q || die 'Could not run git init for hooks.'
fi &&
echo 'Done.'
