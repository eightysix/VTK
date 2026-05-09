# Function to fetch remote modules.

# Helper to perform the initial git clone and checkout.
function(_git_clone git_executable git_repository git_tag module_dir)
  execute_process(
    COMMAND "${git_executable}" clone "${git_repository}" "${module_dir}"
    RESULT_VARIABLE error_code
    OUTPUT_QUIET
    ERROR_VARIABLE git_clone_error
    )
  if(error_code)
    # Parallel configure steps may race on the first clone of a remote module.
    # If another process created the directory first, let update logic handle it.
    if(IS_DIRECTORY "${module_dir}")
      return()
    endif()
    message(FATAL_ERROR "Failed to clone repository: '${git_repository}' Clone error is: '${git_clone_error}'")
  endif()

  execute_process(
    COMMAND "${git_executable}" checkout ${git_tag}
    WORKING_DIRECTORY "${module_dir}"
    RESULT_VARIABLE error_code
    OUTPUT_QUIET
    ERROR_QUIET
    )
  if(error_code)
    message(FATAL_ERROR "Failed to checkout tag: '${git_tag}'")
  endif()

  execute_process(
    COMMAND "${git_executable}" submodule init
    WORKING_DIRECTORY "${module_dir}"
    RESULT_VARIABLE error_code
    )
  if(error_code)
    message(FATAL_ERROR "Failed to init submodules in: '${module_dir}'")
  endif()

  execute_process(
    COMMAND "${git_executable}" submodule update --recursive
    WORKING_DIRECTORY "${module_dir}"
    RESULT_VARIABLE error_code
    )
  if(error_code)
    message(FATAL_ERROR "Failed to update submodules in: '${module_dir}'")
  endif()
endfunction()

# Helper to perform a git update.  Checks the current Git revision against the
# desired revision and only performs a fetch and checkout if needed.
function(_git_update git_executable git_repository git_tag module_dir)
  set(_git_tag_for_fetch "${git_tag}")
  if("${_git_tag_for_fetch}" MATCHES "^origin/(.+)$")
    set(_git_tag_for_fetch "${CMAKE_MATCH_1}")
  endif()

  execute_process(
    COMMAND "${git_executable}" rev-parse --verify "${git_tag}^{commit}"
    WORKING_DIRECTORY "${module_dir}"
    RESULT_VARIABLE error_code
    OUTPUT_VARIABLE tag_hash
    ERROR_VARIABLE git_tag_error
    )
  if(error_code)
    # The ref may not be available yet (e.g., concurrent clone/fetch timing).
    execute_process(
      COMMAND "${git_executable}" fetch origin "${_git_tag_for_fetch}"
      WORKING_DIRECTORY "${module_dir}"
      RESULT_VARIABLE fetch_error
      OUTPUT_QUIET
      ERROR_QUIET
      )
    if(fetch_error)
      execute_process(
        COMMAND "${git_executable}" fetch origin
        WORKING_DIRECTORY "${module_dir}"
        RESULT_VARIABLE fetch_error
        OUTPUT_QUIET
        ERROR_QUIET
        )
    endif()
    if(fetch_error)
      execute_process(
        COMMAND "${git_executable}" fetch "${git_repository}" "${_git_tag_for_fetch}"
        WORKING_DIRECTORY "${module_dir}"
        RESULT_VARIABLE fetch_error
        OUTPUT_QUIET
        ERROR_QUIET
        )
    endif()
    execute_process(
      COMMAND "${git_executable}" rev-parse --verify "${git_tag}^{commit}"
      WORKING_DIRECTORY "${module_dir}"
      RESULT_VARIABLE error_code
      OUTPUT_VARIABLE tag_hash
      ERROR_VARIABLE git_tag_error
      )
    if(error_code AND NOT "${_git_tag_for_fetch}" STREQUAL "${git_tag}")
      execute_process(
        COMMAND "${git_executable}" rev-parse --verify "${_git_tag_for_fetch}^{commit}"
        WORKING_DIRECTORY "${module_dir}"
        RESULT_VARIABLE error_code
        OUTPUT_VARIABLE tag_hash
        ERROR_VARIABLE git_tag_error
        )
    endif()
  endif()
  if(error_code)
    message(FATAL_ERROR "Failed to get the hash for tag '${git_tag}' in '${module_dir}'. Git error is: '${git_tag_error}'")
  endif()
  execute_process(
    COMMAND "${git_executable}" rev-parse --verify HEAD
    WORKING_DIRECTORY "${module_dir}"
    RESULT_VARIABLE error_code
    OUTPUT_VARIABLE head_hash
    )
  if(error_code)
    message(FATAL_ERROR "Failed to get the hash for ${git_repository} HEAD")
  endif()

  # Is the hash checkout out that we want?
  if(NOT (tag_hash STREQUAL head_hash))
    execute_process(
      COMMAND "${git_executable}" fetch origin
      WORKING_DIRECTORY "${module_dir}"
      RESULT_VARIABLE error_code
      )
    if(error_code)
      message(FATAL_ERROR "Failed to fetch repository '${git_repository}'")
    endif()

    execute_process(
      COMMAND "${git_executable}" checkout ${git_tag}
      WORKING_DIRECTORY "${module_dir}"
      RESULT_VARIABLE error_code
      )
    if(error_code AND NOT "${_git_tag_for_fetch}" STREQUAL "${git_tag}")
      execute_process(
        COMMAND "${git_executable}" checkout ${_git_tag_for_fetch}
        WORKING_DIRECTORY "${module_dir}"
        RESULT_VARIABLE error_code
        )
    endif()
    if(error_code)
      message(FATAL_ERROR "Failed to checkout tag: '${git_tag}'")
    endif()

    execute_process(
      COMMAND "${git_executable}" submodule update --recursive
      WORKING_DIRECTORY "${module_dir}"
      RESULT_VARIABLE error_code
      )
    if(error_code)
      message(FATAL_ERROR "Failed to update submodules in: '${module_dir}'")
    endif()
  endif()
endfunction()

# Helper function to fetch a module stored in a Git repository.
# Git fetches are only performed when required.
function(_fetch_with_git git_executable git_repository git_tag module_dir)
  if("${git_tag}" STREQUAL "" OR "${git_repository}" STREQUAL "")
    message(FATAL_ERROR "Tag or repository for git checkout should not be empty.")
  endif()

  # If we have a stale empty directory from an interrupted clone, clean it up.
  if(EXISTS "${module_dir}" AND NOT EXISTS "${module_dir}/.git")
    file(GLOB _module_dir_contents "${module_dir}/*")
    if(NOT _module_dir_contents)
      file(REMOVE_RECURSE "${module_dir}")
    endif()
  endif()

  # If we don't have a clone yet.
  if(NOT EXISTS "${module_dir}")
    _git_clone("${git_executable}" "${git_repository}" "${git_tag}" "${module_dir}")
    if(EXISTS "${module_dir}/.git")
      message(STATUS " The remote module: ${git_repository} is cloned into the directory ${module_dir}")
    endif()
  endif()
  # If another process is cloning this repository concurrently, wait for .git.
  set(_wait_for_git 0)
  while(NOT EXISTS "${module_dir}/.git" AND _wait_for_git LESS 30)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 1)
    math(EXPR _wait_for_git "${_wait_for_git} + 1")
  endwhile()
  if(NOT EXISTS "${module_dir}/.git")
    # Retry once if we ended up with an empty directory.
    if(EXISTS "${module_dir}")
      file(GLOB _module_dir_contents "${module_dir}/*")
      if(NOT _module_dir_contents)
        file(REMOVE_RECURSE "${module_dir}")
        _git_clone("${git_executable}" "${git_repository}" "${git_tag}" "${module_dir}")
      endif()
    endif()
  endif()
  if(NOT EXISTS "${module_dir}/.git")
    message(FATAL_ERROR "Remote module directory '${module_dir}' exists, but it is not a valid git checkout.")
  endif()
  # We already have a clone (or a concurrent configure step created it), so check revision.
  _git_update("${git_executable}" "${git_repository}" "${git_tag}" "${module_dir}")
endfunction()

# Download and turn on a remote module.
#
# The CMake variable VTK_MODULE_ENABLE_VTK_${module_name} is created
# if it does not exist. The variable is set to DEFAULT.
# Once set to WANT or YES, the module is downloaded into the Remote module
# group.
#
# A module name and description are required.  The description will
# show up in the CMake user interface.
#
# The following options are currently supported:
#    [GIT_REPOSITORY url]        # URL of git repo
#    [GIT_TAG tag]               # Git branch name, commit id or tag
function(vtk_fetch_module _name _description)
  # If the variable does not exist, create it and set its value to
  # DEFAULT.
  if(NOT VTK_MODULE_ENABLE_VTK_${_name})
    set("VTK_MODULE_ENABLE_VTK_${_name}" "DEFAULT"
      CACHE STRING "Enable the ${name} module. ${_description}")
    mark_as_advanced("VTK_MODULE_ENABLE_VTK_${_name}")
    set_property(CACHE "VTK_MODULE_ENABLE_VTK_${_name}"
      PROPERTY
      STRINGS "YES;WANT;DONT_WANT;NO;DEFAULT")
  endif()
  # If the variable is WANT or YES, download the module.
  if (VTK_MODULE_ENABLE_VTK_${_name} STREQUAL "WANT" OR
      VTK_MODULE_ENABLE_VTK_${_name} STREQUAL "YES" OR
      (VTK_MODULE_ENABLE_VTK_${_name} STREQUAL "DEFAULT" AND VTK_BUILD_ALL_MODULES))
    vtk_download_attempt_check(Module_${_name})
    cmake_parse_arguments(PARSE_ARGV 2 _fetch_options "" "GIT_REPOSITORY;GIT_TAG" "")
    find_package(Git)
    if(NOT GIT_EXECUTABLE)
      message(FATAL_ERROR "error: could not find git for clone of ${_name}")
    endif()
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" --version
      OUTPUT_VARIABLE ov
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    if(GIT_VERSION_STRING VERSION_LESS 1.6.6)
      message(FATAL_ERROR "Git version 1.6.6 or later is required.")
    endif()
    _fetch_with_git("${GIT_EXECUTABLE}"
      "${_fetch_options_GIT_REPOSITORY}"
      "${_fetch_options_GIT_TAG}"
      "${VTK_SOURCE_DIR}/Remote/${_name}"
      )
  endif()
endfunction()
