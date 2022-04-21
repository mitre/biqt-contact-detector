# Windows

Windows is not a supported build target for this provider.

# Linux (CentOS Linux 7.9)

## Prerequisites

This guide targets CentOS Linux 7.9.

This provider relies on core development tools including gcc. These dependencies can be installed from 
the [Extra Packages for Enterprise Linux (EPEL)](https://fedoraproject.org/wiki/EPEL#How_can_I_use_these_extra_packages.3F) 
repository using the following commands:

```bash
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake3
```

This provider also relies on python 3.7. The source can be obtained via https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz and compiled:

```
yum install gcc openssl-devel bzip2-devel libffi-devel zlib-devel wget -y
mkdir /usr/src 
cd /usr/src
wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz
tar xzf Python-3.7.9.tgz
cd Python-3.7.9
./configure --enable-shared --enable-optimizations
make -j$(nproc) altinstall
rm -rf /usr/src/Python-3.7.9.tgz /usr/src/Python-3.7.9/
```

Using make altinstall will install to directories called **python3.7**, preventing overwriting
the contents of original python directories.

## Building and Installing

### Building the Provider

Clone the repository.

```bash
git clone https://github.com/mitre/biqt-contact-detector.git
cd biqt-contact-detector
```

Download the python dependencies.

```
pip3.7 install -r requirements.txt
```

Finally, build and install the provider.

```
mkdir build
cd build
cmake3 -DCMAKE_BUILD_TYPE=Release ..
make
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
