FROM crzynik/swatch-nginx:latest as nginx

# Pinned to bookworm rather than the floating `python:3` tag: that tag has
# already drifted to Debian trixie, which dropped the libpcre3 package
# entirely (Debian's phasing out legacy PCRE1 in favor of PCRE2) -- but the
# crzynik/swatch-nginx binary above is linked against libpcre.so.3, so it
# needs a Debian release that still ships it. Pinning avoids this breaking
# again on a future rebuild if the tag drifts further.
FROM python:3-bookworm

# install dependencies
COPY requirements.txt /requirements.txt
COPY requirements_dev.txt /requirements_dev.txt

ENV OPENCV_VERSION="4.5.1"
ENV FLASK_ENV=development

# ffmpeg is used to pull audio from camera RTSP streams for audio_monitors.
# libpcre3 provides libpcre.so.3, which the nginx binary above needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libpcre3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

COPY --from=nginx /usr/local/nginx/ /usr/local/nginx/

# build swatch frontend
WORKDIR /opt/swatch/

COPY swatch swatch/
COPY web/build/web frontend/
COPY migrations migrations/

COPY docker/rootfs /

# general docker

EXPOSE 4500

CMD ["python3", "-u", "-m", "swatch"]
