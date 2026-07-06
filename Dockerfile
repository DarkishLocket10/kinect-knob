# kinect-knob — hand-gesture volume knob for Home Assistant
#
# Multi-stage build:
#   build stage    compiles libfreenect v0.7.5 (Kinect v1, Xbox 360) with its
#                  Python 3 binding, and libfreenect2 v0.2.1 (Kinect v2, Xbox
#                  One) headless with the OpenCL depth packet processor, plus
#                  a wheel of the cffi 'freenect2' Python binding.
#   runtime stage  slim Ubuntu 22.04 (Python 3.10) with only runtime libs.
#
# GPU: Kinect v2 depth decode uses OpenCL on the NVIDIA GPU via the NVIDIA
# container runtime (run with --runtime=nvidia). No CUDA toolkit needed —
# OpenCL decodes a depth frame in ~1 ms on a GTX 1080 Ti. Kinect v1 needs no
# GPU at all. Hand tracking itself runs on CPU (fastest option per MediaPipe).

# ---------------------------------------------------------------------------
FROM ubuntu:22.04 AS build
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config git ca-certificates \
    libusb-1.0-0-dev libturbojpeg0-dev \
    opencl-headers ocl-icd-opencl-dev \
    python3 python3-dev python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Cython >=0.29.31 required by libfreenect v0.7.5's python wrapper; numpy
# headers here determine the ABI the wrapper is compiled against (numpy 2).
RUN pip3 install --no-cache-dir "cython>=3.0" "numpy>=2.0,<3" "cffi>=1.16"

# --- Kinect v1: libfreenect + python3 binding ------------------------------
RUN git clone --depth 1 --branch v0.7.5 https://github.com/OpenKinect/libfreenect.git /opt/libfreenect \
 && cmake -S /opt/libfreenect -B /opt/libfreenect/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_PYTHON3=ON \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_FAKENECT=OFF \
      -DBUILD_OPENNI2_DRIVER=OFF \
 && cmake --build /opt/libfreenect/build -j"$(nproc)" \
 && cmake --install /opt/libfreenect/build

# --- Kinect v2: libfreenect2, headless, OpenCL depth processor -------------
RUN git clone --depth 1 --branch v0.2.1 https://github.com/OpenKinect/libfreenect2.git /opt/libfreenect2 \
 && cmake -S /opt/libfreenect2 -B /opt/libfreenect2/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DENABLE_OPENGL=OFF \
      -DENABLE_OPENCL=ON \
      -DENABLE_CUDA=OFF \
      -DBUILD_OPENNI2_DRIVER=OFF \
      -DBUILD_EXAMPLES=OFF \
 && cmake --build /opt/libfreenect2/build -j"$(nproc)" \
 && cmake --install /opt/libfreenect2/build

# --- Kinect v2 python binding (cffi; finds libfreenect2 via pkg-config) ----
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
RUN pip3 wheel --no-deps --wheel-dir /wheels freenect2==0.2.3

# ---------------------------------------------------------------------------
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # NVIDIA container runtime: inject compute (CUDA/OpenCL) + utility libs.
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    # Force libfreenect2's OpenCL depth pipeline (cl | cuda | cpu).
    LIBFREENECT2_PIPELINE=cl \
    KK_MODEL_PATH=/app/models/hand_landmarker.task

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libusb-1.0-0 libturbojpeg \
    ocl-icd-libopencl1 clinfo \
    libgl1 libglib2.0-0 \
    libegl1 libgles2 \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# OpenCL ICD: the NVIDIA runtime injects libnvidia-opencl.so.1; this file
# tells the ICD loader to use it. Without it, clinfo sees 0 platforms and
# libfreenect2 silently falls back to the (too slow) CPU pipeline.
RUN mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Compiled Kinect stacks (libs + the cython 'freenect' module in dist-packages).
COPY --from=build /usr/local /usr/local
RUN ldconfig

WORKDIR /app
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY --from=build /wheels /tmp/wheels
RUN pip3 install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Bake the hand landmark model into the image (no internet needed at runtime).
RUN mkdir -p /app/models && curl -fsSL -o /app/models/hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

COPY pyproject.toml README.md ./
COPY src ./src
# Ubuntu 22.04 ships setuptools 59.6, which predates PEP 621 ([project] table)
# support — installing without upgrading silently builds an empty "UNKNOWN"
# package and the kinectknob module never lands in site-packages.
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel \
 && pip3 install --no-cache-dir --no-deps .

EXPOSE 8420
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD curl -sf "http://localhost:${KK_PORT:-8420}/healthz" || exit 1

ENTRYPOINT ["python3", "-m", "kinectknob"]
