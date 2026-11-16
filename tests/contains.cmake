# SPDX-License-Identifier: MIT

# cmake -DFILE=<path> -DNEEDLE=<text> -P contains.cmake: fails unless FILE contains NEEDLE.
file(READ "${FILE}" content)
string(FIND "${content}" "${NEEDLE}" at)
if(at EQUAL -1)
  message(FATAL_ERROR "${FILE} does not mention ${NEEDLE}")
endif()
