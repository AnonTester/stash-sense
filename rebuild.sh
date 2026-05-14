#!/bin/sh
docker compose -f docker-stashsense.yml down
docker compose -f docker-stashsense.yml up -d --force-recreate --build
