#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file refered to github.com/onnx/onnx.git

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.spawn import find_executable
from distutils import sysconfig, log
import setuptools
import setuptools.command.build_py
import setuptools.command.develop
import setuptools.command.build_ext

from collections import namedtuple
from contextlib import contextmanager
import glob
import os
import shlex
import subprocess
import sys
import platform
from textwrap import dedent
import multiprocessing

with open("requirements.txt") as fin:
    REQUIRED_PACKAGES = fin.read()

PACKAGE_NAME = "fastdeploy"
setup_configs = dict()
setup_configs["ENABLE_PADDLE_FRONTEND"] = os.getenv("ENABLE_PADDLE_FRONTEND",
                                                    "ON")
setup_configs["ENABLE_ORT_BACKEND"] = os.getenv("ENABLE_ORT_BACKEND", "ON")
setup_configs["ENABLE_PADDLE_BACKEND"] = os.getenv("ENABLE_PADDLE_BACKEND",
                                                   "OFF")
setup_configs["BUILD_DEMO"] = os.getenv("BUILD_DEMO", "ON")
setup_configs["ENABLE_VISION"] = os.getenv("ENABLE_VISION", "ON")
setup_configs["ENABLE_TRT_BACKEND"] = os.getenv("ENABLE_TRT_BACKEND", "OFF")
setup_configs["WITH_GPU"] = os.getenv("WITH_GPU", "OFF")
setup_configs["TRT_DIRECTORY"] = os.getenv("TRT_DIRECTORY", "UNDEFINED")
setup_configs["CUDA_DIRECTORY"] = os.getenv("CUDA_DIRECTORY",
                                            "/usr/local/cuda")
if os.getenv("CMAKE_CXX_COMPILER", None) is not None:
    setup_configs["CMAKE_CXX_COMPILER"] = os.getenv("CMAKE_CXX_COMPILER")

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOP_DIR, "fastdeploy")
CMAKE_BUILD_DIR = os.path.join(TOP_DIR, '.setuptools-cmake-build')

WINDOWS = (os.name == 'nt')

CMAKE = find_executable('cmake3') or find_executable('cmake')
MAKE = find_executable('make')

setup_requires = []
extras_require = {}

################################################################################
# Global variables for controlling the build variant
################################################################################

# Default value is set to TRUE\1 to keep the settings same as the current ones.
# However going forward the recomemded way to is to set this to False\0
USE_MSVC_STATIC_RUNTIME = bool(
    os.getenv('USE_MSVC_STATIC_RUNTIME', '1') == '1')
ONNX_NAMESPACE = os.getenv('ONNX_NAMESPACE', 'paddle2onnx')
################################################################################
# Version
################################################################################

try:
    git_version = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=TOP_DIR).decode('ascii').strip()
except (OSError, subprocess.CalledProcessError):
    git_version = None

with open(os.path.join(TOP_DIR, 'VERSION_NUMBER')) as version_file:
    VersionInfo = namedtuple('VersionInfo', ['version', 'git_version'])(
        version=version_file.read().strip(), git_version=git_version)

################################################################################
# Pre Check
################################################################################

assert CMAKE, 'Could not find "cmake" executable!'

################################################################################
# Utilities
################################################################################


@contextmanager
def cd(path):
    if not os.path.isabs(path):
        raise RuntimeError('Can only cd to absolute path, got: {}'.format(
            path))
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


################################################################################
# Customized commands
################################################################################


class ONNXCommand(setuptools.Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass


def get_all_files(dirname):
    files = list()
    for root, dirs, filenames in os.walk(dirname):
        for f in filenames:
            fullname = os.path.join(root, f)
            files.append(fullname)
    return files


class create_version(ONNXCommand):
    def run(self):
        with open(os.path.join(SRC_DIR, 'version.py'), 'w') as f:
            f.write(
                dedent('''\
            # This file is generated by setup.py. DO NOT EDIT!
            from __future__ import absolute_import
            from __future__ import division
            from __future__ import print_function
            from __future__ import unicode_literals
            version = '{version}'
            git_version = '{git_version}'
            '''.format(**dict(VersionInfo._asdict()))))


class cmake_build(setuptools.Command):
    """
    Compiles everything when `python setupmnm.py build` is run using cmake.
    Custom args can be passed to cmake by specifying the `CMAKE_ARGS`
    environment variable.
    The number of CPUs used by `make` can be specified by passing `-j<ncpus>`
    to `setup.py build`.  By default all CPUs are used.
    """
    user_options = [(str('jobs='), str('j'),
                     str('Specifies the number of jobs to use with make'))]

    built = False

    def initialize_options(self):
        self.jobs = None

    def finalize_options(self):
        if sys.version_info[0] >= 3:
            self.set_undefined_options('build', ('parallel', 'jobs'))
        if self.jobs is None and os.getenv("MAX_JOBS") is not None:
            self.jobs = os.getenv("MAX_JOBS")
        self.jobs = multiprocessing.cpu_count() if self.jobs is None else int(
            self.jobs)

    def run(self):
        if cmake_build.built:
            return
        cmake_build.built = True
        if not os.path.exists(CMAKE_BUILD_DIR):
            os.makedirs(CMAKE_BUILD_DIR)

        with cd(CMAKE_BUILD_DIR):
            build_type = 'Release'
            # configure
            cmake_args = [
                CMAKE,
                '-DPYTHON_INCLUDE_DIR={}'.format(sysconfig.get_python_inc()),
                '-DPYTHON_EXECUTABLE={}'.format(sys.executable),
                '-DBUILD_FASTDEPLOY_PYTHON=ON',
                '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                '-DONNX_NAMESPACE={}'.format(ONNX_NAMESPACE),
                '-DPY_EXT_SUFFIX={}'.format(
                    sysconfig.get_config_var('EXT_SUFFIX') or ''),
            ]
            cmake_args.append('-DCMAKE_BUILD_TYPE=%s' % build_type)
            for k, v in setup_configs.items():
                cmake_args.append("-D{}={}".format(k, v))
            if WINDOWS:
                cmake_args.extend([
                    # we need to link with libpython on windows, so
                    # passing python version to window in order to
                    # find python in cmake
                    '-DPY_VERSION={}'.format('{0}.{1}'.format(* \
                                                              sys.version_info[:2])),
                ])
                if platform.architecture()[0] == '64bit':
                    cmake_args.extend(['-A', 'x64', '-T', 'host=x64'])
                else:
                    cmake_args.extend(['-A', 'Win32', '-T', 'host=x86'])
            if 'CMAKE_ARGS' in os.environ:
                extra_cmake_args = shlex.split(os.environ['CMAKE_ARGS'])
                # prevent crossfire with downstream scripts
                del os.environ['CMAKE_ARGS']
                log.info('Extra cmake args: {}'.format(extra_cmake_args))
                cmake_args.extend(extra_cmake_args)
            cmake_args.append(TOP_DIR)
            subprocess.check_call(cmake_args)

            build_args = [CMAKE, '--build', os.curdir]
            if WINDOWS:
                build_args.extend(['--config', build_type])
                build_args.extend(['--', '/maxcpucount:{}'.format(self.jobs)])
            else:
                build_args.extend(['--', '-j', str(self.jobs)])
            subprocess.check_call(build_args)


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command('create_version')
        self.run_command('cmake_build')

        generated_python_files = \
            glob.glob(os.path.join(CMAKE_BUILD_DIR, 'fastdeploy', '*.py')) + \
            glob.glob(os.path.join(CMAKE_BUILD_DIR, 'fastdeploy', '*.pyi'))

        for src in generated_python_files:
            dst = os.path.join(TOP_DIR, os.path.relpath(src, CMAKE_BUILD_DIR))
            self.copy_file(src, dst)

        return setuptools.command.build_py.build_py.run(self)


class develop(setuptools.command.develop.develop):
    def run(self):
        self.run_command('build_py')
        setuptools.command.develop.develop.run(self)


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        self.run_command('cmake_build')
        setuptools.command.build_ext.build_ext.run(self)

    def build_extensions(self):
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = os.path.basename(self.get_ext_filename(fullname))

            lib_path = CMAKE_BUILD_DIR
            if os.name == 'nt':
                debug_lib_dir = os.path.join(lib_path, "Debug")
                release_lib_dir = os.path.join(lib_path, "Release")
                if os.path.exists(debug_lib_dir):
                    lib_path = debug_lib_dir
                elif os.path.exists(release_lib_dir):
                    lib_path = release_lib_dir
            src = os.path.join(lib_path, filename)
            dst = os.path.join(
                os.path.realpath(self.build_lib), "fastdeploy", filename)
            self.copy_file(src, dst)


class mypy_type_check(ONNXCommand):
    description = 'Run MyPy type checker'

    def run(self):
        """Run command."""
        onnx_script = os.path.realpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "tools/mypy-onnx.py"))
        returncode = subprocess.call([sys.executable, onnx_script])
        sys.exit(returncode)


cmdclass = {
    'create_version': create_version,
    'cmake_build': cmake_build,
    'build_py': build_py,
    'develop': develop,
    'build_ext': build_ext,
    'typecheck': mypy_type_check,
}

################################################################################
# Extensions
################################################################################

ext_modules = [
    setuptools.Extension(
        name=str(PACKAGE_NAME + '.fastdeploy_main'), sources=[]),
]

################################################################################
# Packages
################################################################################

# no need to do fancy stuff so far
packages = setuptools.find_packages()

################################################################################
# Test
################################################################################

if sys.version_info[0] == 3:
    # Mypy doesn't work with Python 2
    extras_require['mypy'] = ['mypy==0.600']

################################################################################
# Final
################################################################################

package_data = {PACKAGE_NAME: ["LICENSE", "ThirdPartyNotices.txt"]}

if sys.argv[1] == "install" or sys.argv[1] == "bdist_wheel":
    if not os.path.exists(".setuptools-cmake-build"):
        print("Please execute `python setup.py build` first.")
        sys.exit(0)
    import shutil

    shutil.copy("ThirdPartyNotices.txt", "fastdeploy")
    shutil.copy("LICENSE", "fastdeploy")
    depend_libs = list()

    # copy fastdeploy library
    pybind_so_file = None
    for f in os.listdir(".setuptools-cmake-build"):
        if not os.path.isfile(os.path.join(".setuptools-cmake-build", f)):
            continue
        if f.count("fastdeploy") > 0:
            shutil.copy(
                os.path.join(".setuptools-cmake-build", f), "fastdeploy/libs")
        if f.count("fastdeploy_main.cpython-"):
            pybind_so_file = os.path.join(".setuptools-cmake-build", f)

    if not os.path.exists(".setuptools-cmake-build/third_libs/install"):
        raise Exception(
            "Cannot find directory third_libs/install in .setuptools-cmake-build."
        )

    if os.path.exists("fastdeploy/libs/third_libs"):
        shutil.rmtree("fastdeploy/libs/third_libs")
    shutil.copytree(
        ".setuptools-cmake-build/third_libs/install",
        "fastdeploy/libs/third_libs",
        symlinks=True)

    if platform.system().lower() == "windows":
        release_dir = os.path.join(".setuptools-cmake-build", "Release")
        for f in os.listdir(release_dir):
            filename = os.path.join(release_dir, f)
            if not os.path.isfile(filename):
                continue
            if filename.endswith(".pyd"):
                continue
            shutil.copy(filename, "fastdeploy/libs")

    if platform.system().lower() == "linux":
        rpaths = ["$ORIGIN:$ORIGIN/libs"]
        for root, dirs, files in os.walk(
                ".setuptools-cmake-build/third_libs/install"):
            for d in dirs:
                if d == "lib":
                    path = os.path.relpath(
                        os.path.join(root, d),
                        ".setuptools-cmake-build/third_libs/install")
                    rpaths.append("$ORIGIN/" + os.path.join("libs/third_libs",
                                                            path))
        rpaths = ":".join(rpaths)
        command = "patchelf --set-rpath '{}' ".format(rpaths) + pybind_so_file
        # The sw_64 not suppot patchelf, so we just disable that.
        if platform.machine() != 'sw_64' and platform.machine() != 'mips64':
            assert os.system(
                command) == 0, "patchelf {} failed, the command: {}".format(
                    command, pybind_so_file)
    elif platform.system().lower() == "darwin":
        pre_commands = [
            "install_name_tool -delete_rpath '@loader_path/libs' " +
            pybind_so_file
        ]
        commands = [
            "install_name_tool -id '@loader_path/libs' " + pybind_so_file
        ]
        commands.append("install_name_tool -add_rpath '@loader_path/libs' " +
                        pybind_so_file)
        for root, dirs, files in os.walk(
                ".setuptools-cmake-build/third_libs/install"):
            for d in dirs:
                if d == "lib":
                    path = os.path.relpath(
                        os.path.join(root, d),
                        ".setuptools-cmake-build/third_libs/install")
                    pre_commands.append(
                        "install_name_tool -delete_rpath '@loader_path/{}' ".
                        format(os.path.join("libs/third_libs",
                                            path)) + pybind_so_file)
                    commands.append(
                        "install_name_tool -add_rpath '@loader_path/{}' ".
                        format(os.path.join("libs/third_libs",
                                            path)) + pybind_so_file)
        for command in pre_commands:
            try:
                os.system(command)
            except:
                print("Skip execute command: " + command)
        for command in commands:
            assert os.system(
                command) == 0, "command execute failed! command: {}".format(
                    command)

    all_files = get_all_files("fastdeploy/libs")
    for f in all_files:
        package_data[PACKAGE_NAME].append(os.path.relpath(f, "fastdeploy"))

setuptools.setup(
    name=PACKAGE_NAME,
    version=VersionInfo.version,
    description="Deploy Kit Tool For Deeplearning models.",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=packages,
    package_data=package_data,
    include_package_data=True,
    setup_requires=setup_requires,
    extras_require=extras_require,
    author='fastdeploy',
    author_email='fastdeploy@baidu.com',
    url='https://github.com/PaddlePaddle/FastDeploy.git',
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0')
