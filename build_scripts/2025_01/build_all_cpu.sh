#!/bin/bash

# Builds the following wind tools: ROSCO, OpenFAST, AMR-Wind, and ERF
# Regis Thedin, 2025-01

# --------------------------- USER INPUT ------------------------------ #
build_rosco=false
build_openfast_standalone=false
build_amrwind_with_openfast=false
build_amrwind_standalone=false
build_erf=true

tag_rosco='v2.9.0'
tag_openfast='v3.5.5'
tag_amrwind='v3.3.1'
tag_erf='25.01'
# --------------------------------------------------------------------- #

# Base directories
basedir=${PWD}
scriptdir='/home/rthedin/utilities/build_scripts/2025_01'

# Valid releases/tags
valid_tag_rosco=('main' 'develop' 'v2.7.1' 'v2.8.1' 'v2.9.0')
valid_tag_openfast=('main' 'dev' 'rc-3.5.5' 'v3.5.5' 'v4.0.0')
valid_tag_amrwind=('main' 'v3.0.2' 'v3.1.7' 'v3.2.0' 'v3.3.0' 'v3.3.1')
valid_tag_erf=('development' '24.11' '24.10' '24.09' '25.01')

is_valid_tag() {
    local tag=$1; shift
    [[ " ${@} " =~ " ${tag} " ]]
}

if [[ $build_amrwind_with_openfast == true && $tag_openfast == "v4.0.0" ]]; then
    echo "ERROR: AMR-Wind does not support OpenFAST 4.0.0. Standalone OpenFAST is available"
    exit 1
fi

if [[ $build_amrwind_with_openfast == true ]]; then
    build_openfast_standalone=true
    build_amrwind_standalone=false
fi

# ---------- Check if git repository already exists and build directory exists
if [[ $build_rosco == true ]]; then
	if [[ $tag_rosco == "v2.7.1" || $tag_rosco == "v2.8.1" ]]; then
        file=$basedir/ROSCO_$tag_rosco/ROSCO/build/libdiscon.so
	else  # 2.9.0 onwards
        file=$basedir/ROSCO_$tag_rosco/rosco/controller/build/libdiscon.so
	fi
	if [ -f $file ]; then
        echo "ROSCO $tag_rosco already built. Skipping it."
        build_rosco=false
    fi
fi

if [[ $build_openfast_standalone == true ]]; then
    file=$basedir/openfast_$tag_openfast/build/install/bin/openfast
	if [ -f $file ]; then
	    echo "OpenFAST $tag_openfast already built. Skipping it."
        build_openfast_standalone=false
	fi
    export openfastpath=$basedir/openfast_$tag_openfast/build/install
fi

if [[ $build_amrwind_with_openfast == true ]]; then
    file=$basedir/amr-wind_$tag_amrwind/build_with_openfast_$tag_openfast/install/bin/amr_wind
    if [ -f $file ]; then
	    echo "AMR-Wind $tag_amrwind with OpenFAST $tag_openfast already built. Skipping it."
        build_amrwind_with_openfast=false
    fi

elif [[ $build_amrwind_standalone == true ]]; then
    file=$basedir/amr-wind_$tag_amrwind/build_standalone/bin/amr_wind
    if [ -f $file ]; then
        echo "AMR-Wind $tag_amrwind (standalone) already built. Skipping it."
        build_amrwind_standalone=false
	fi

fi

if [[ $build_erf == true ]]; then
    file=$basedir/ERF_$tag_erf/build/install/bin/erf_abl
    if [ -f $file ]; then
        echo "ERF $tag_erf already built. Skipping it."
        build_erf=false
	fi
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

if [[ $build_openfast_standalone == true ]]; then
	if is_valid_tag "$tag_openfast" "${valid_tag_openfast[@]}"; then
		echo "Cloning OpenFAST $tag_openfast"
		git clone -q --depth 1 --branch $tag_openfast git@github.com:OpenFAST/openfast.git openfast_$tag_openfast > /dev/null 2>&1
	else
		echo "ERROR: OpenFAST tag $tag_openfast not valid. Valid options are ${valid_tag_openfast[@]}."
		exit 1
	fi
fi

if [[ $build_amrwind_with_openfast == true || $build_amrwind_standalone == true ]]; then
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

if [[ $build_openfast_standalone == true ]]; then
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
	# Build AMR-Wind with openfast
	echo "Building AMR-Wind $tag_amrwind with OpenFAST $tag_openfast..."
	cd $basedir/amr-wind_$tag_amrwind
	mkdir build_with_openfast_$tag_openfast; cd build_with_openfast_$tag_openfast
	cp $scriptdir/build_amrwind_${tag_amrwind}_openfast_${tag_openfast}_cpu_stalllibs.sh .
	./build_amrwind_${tag_amrwind}_openfast_${tag_openfast}_cpu_stalllibs.sh > log.build.amrwind_$tag_amrwind.openfast_$tag_openfast 2>&1
	if [ $? -eq 0 ]; then
	  echo "Successfully built AMR-Wind. Build log located at ${PWD}/log.build.amrwind_$tag_amrwind.openfast_$tag_openfast"
	else
	  echo "AMR-Wind build failed. Build log located at ${PWD}/log.build.amrwind_$tag_amrwind.openfast_$tag_openfast"
	fi
elif [[ $build_amrwind_standalone == true ]]; then
    # Build AMR-Wind (standalone, without openfast)
	echo "Building AMR-Wind $tag_amrwind..."
	cd $basedir/amr-wind_$tag_amrwind
	mkdir build_standalone; cd build_standalone
	cp $scriptdir/build_amrwind_${tag_amrwind}_cpu_stalllibs.sh .
	./build_amrwind_${tag_amrwind}_cpu_stalllibs.sh > log.build.amrwind_$tag_amrwind 2>&1
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


