#!/usr/bin/env bash

# if config folder does not exist, or force env var is true, then create symlinks
if [ ! -e ~/.config ] || [ "$FORCE" = "1" ]; then
    cd ~

    for p in .local .ssh .config .ipython .jupyter .git-credentials .gitconfig .bash_history .netrc;
    do
        if [ -e /storage/cfg/$p ]; then
            rm -rf $p
            ln -s /storage/cfg/$p
        fi
    done
fi

if [ ! -e /notebooks/git ]; then
  if [ ! -e /storage/git ]; then
      mkdir /storage/git
  fi
  ln -s /storage/git/ /notebooks/
fi

# enable ssh-agent
eval $(ssh-agent -s)