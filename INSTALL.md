# Windows

Windows is not a supported build target for this provider.

# Linux (Ubuntu Linux 22.04)

## Prerequisites

This guide targets Ubuntu Linux 22.04.

This provider relies on core development tools including gcc. These dependencies can be installed using the following commands:

```bash
apt update
apt install -y cmake python3 python3-pip
```

## Building and Installing

### Building the Provider

Clone the repository.

```bash
git clone https://github.com/mitre/biqt-contact-detector.git
cd biqt-contact-detector
```

Download the python dependencies for GPU build.

```
pip install -r requirements.txt
```

Download the python dependencies for CPU build;

```
pip install -r requirements-cpu.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

Finally, build and install the provider. Set BIQT_HOME via environment variable or using -DBIQT_HOME in cmake command below.

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
sudo make install
```

## Running the Provider

After installation, you can call the provider using the reference BIQT CLI as follows:

```
# Runs all iris providers (including this one).
$>biqt -m "iris" <image path>

# Runs only the biqt-contact-detector provider.
$>biqt -p BIQTContactDetector <image path>
```
