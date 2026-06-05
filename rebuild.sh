#!/bin/sh
docker compose -f docker-stashsense.yml down
docker compose -f docker-stashsense.yml up -d --force-recreate --build
cp plugin/* /opt/stash-storage/config/plugins/stash-sense --preserve 2>&1 | grep -v 'plugin/__pycache__' 

