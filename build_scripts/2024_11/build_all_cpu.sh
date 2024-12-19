#!/bin/bash

# Builds the following wind tools: ROSCO, OpenFAST, AMR-Wind, and ERF
# Regis Thedin, 2024-11-15

# --------------------------- USER INPUT ------------------------------ #
build_rosco=true
build_openfast=true
build_amrwind_with_openfast=true
build_erf=true

tag_rosco='v2.9.0'
tag_openfast='rc-3.5.5'
tag_amrwind='v3.2.0'
tag_erf='24.11'
# --------------------------------------------------------------------- #

# Base directories
basedir=${PWD}
scriptdir='/home/rthedin/utilities/build_scripts/2024_11'

# Valid releases/tags
valid_tag_rosco=('main' 'develop' 'v2.7.1' 'v2.8.1' 'v2.9.0')
valid_tag_openfast=('main' 'dev' 'rc-3.5.5')
valid_tag_amrwind=('main' 'v3.0.2' 'v3.1.7' 'v3.2.0')
valid_tag_erf=('development' '24.11' '24.10' '24.09')

is_valid_tag() {
    local tag=$1; shift
    [[ " ${@} " =~ " ${tag} " ]]
}

if [[ $build_amrwind_with_openfast == true ]]; then
    build_openfast=true
fi

# ---------- Check requested version and clone repositories
if [[ $build_rosco == true ]]; then
	if is_valid_tag "$tag_rosco" "${valid_tag_rosco[@]}"; then
		echo "Cloning ROSCO $tag_rosco"
		git clone -q --depth 1 --branch $tag_rosco git@github.com:NREL/ROSCO.git ROSCO_$tag_rosco > /dev/null 2>&1
	else
		echo "ERROR: ROSCO tag $tag_rosco not valid. Valid options are ${valid_tag_rosco[@]}."
		exit 1
	fi
fi

if [[ $build_openfast == true ]]; then
	if is_valid_tag "$tag_openfast" "${valid_tag_openfast[@]}"; then
		echo "Cloning OpenFAST $tag_openfast"
		git clone -q --depth 1 --branch $tag_openfast git@github.com:OpenFAST/openfast.git openfast_$tag_openfast > /dev/null 2>&1
	else
		echo "ERROR: OpenFAST tag $tag_openfast not valid. Valid options are ${valid_tag_openfast[@]}."
		exit 1
	fi
fi

if [[ $build_amrwind_with_openfast == true ]]; then
	if is_valid_tag "$tag_amrwind" "${valid_tag_amrwind[@]}"; then
		echo "Cloning AMR-Wind $tag_amrwind"
		git clone -q --recursive --depth 1 --branch $tag_amrwind git@github.com:Exawind/amr-wind.git amr-wind_$tag_amrwind > /dev/null 2>&1
	else
		echo "ERROR: AMR-Wind tag $tag_amrwind not valid. Valid options are ${valid_tag_amrwind[@]}."
		exit 1
	fi
fi

if [[ $build_erf == true ]]; then
	if is_valid_tag "$tag_erf" "${valid_tag_erf[@]}"; then
		echo "Cloning ERF $tag_erf"
		git clone -q --recursive --depth 1 --branch $tag_erf git@github.com:erf-model/ERF.git ERF_$tag_erf > /dev/null 2>&1
	else
		echo "ERROR: ERF tag $tag_erf not valid. Valid options are ${valid_tag_erf[@]}."
		exit 1
	fi
fi

# ---------- Build tools
if [[ $build_rosco == true ]]; then
	# Build ROSCO
	echo "Building ROSCO $tag_rosco..."
	if [[ $tag_rosco == "v2.7.1" || $tag_rosco == "v2.8.1" ]]; then
		cd $basedir/ROSCO_$tag_rosco/ROSCO
	else  # 2.9.0 onwards
		cd $basedir/ROSCO_$tag_rosco/rosco/controller
	fi
    mkdir build; cd build
    cp $scriptdir/build_rosco_$tag_rosco.sh .
	./build_rosco_$tag_rosco.sh > log.build.rosco_$tag_rosco 2>&1
	if [ $? -eq 0 ]; then
	  echo "Successfully built ROSCO. Build log located at ${PWD}/log.build.rosco_$tag_rosco"
	else
	  echo "ROSCO build failed. Build log located at ${PWD}/log.build.rosco_$tag_rosco"
	fi
fi

if [[ $build_openfast == true ]]; then
	# Build OpenFAST
	echo "Building OpenFAST $tag_openfast..."
	cd $basedir/openfast_$tag_openfast
	mkdir build; cd build
	cp $scriptdir/build_openfast_$tag_openfast.sh .
	./build_openfast_$tag_openfast.sh > log.build.openfast_$tag_openfast 2>&1
	if [ $? -eq 0 ]; then
	  echo "Successfully built OpenFAST. Build log located at ${PWD}/log.build.openfast_$tag_openfast"
	else
	  echo "OpenFAST build failed. Build log located at ${PWD}/log.build.openfast_$tag_openfast"
	fi
	export openfastpath=${PWD}/install
fi

if [[ $build_amrwind_with_openfast == true ]]; then
	# Build AMR-Wind
	echo "Building AMR-Wind $tag_amrwind..."
	cd $basedir/amr-wind_$tag_amrwind
	mkdir build_with_openfast_$tag_openfast; cd build_with_openfast_$tag_openfast
	cp $scriptdir/build_amrwind_${tag_amrwind}_openfast_${tag_openfast}_cpu_stalllibs.sh .
	./build_amrwind_${tag_amrwind}_openfast_${tag_openfast}_cpu_stalllibs.sh > log.build.amrwind_$tag_amrwind 2>&1
	if [ $? -eq 0 ]; then
	  echo "Successfully built AMR-Wind. Build log located at ${PWD}/log.build.amrwind_$tag_amrwind"
	else
	  echo "AMR-Wind build failed. Build log located at ${PWD}/log.build.amrwind_$tag_amrwind"
	fi
fi

if [[ $build_erf == true ]]; then
	# Build ERF
	echo "Building ERF $tag_erf..."
	cd $basedir/ERF_$tag_erf
	mkdir build; cd build
	cp $scriptdir/build_erf_$tag_erf.sh .
	./build_erf_$tag_erf.sh > log.build_erf_$tag_erf 2>&1
	if [ $? -eq 0 ]; then
	  echo "Successfully built ERF. Build log located at ${PWD}/log.build.build_erf_$tag_erf"
	else
	  echo "ERF build failed. Build log located at ${PWD}/log.build.build_erf_$tag_erf"
	fi
fi

cd $basedir


