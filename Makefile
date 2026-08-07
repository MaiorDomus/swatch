default_target: local

local:
	cd web; flutter build web;
	DOCKER_BUILDKIT=1 docker build --no-cache -t swatch -f docker/Dockerfile .
