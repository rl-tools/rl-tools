# Embeds a text file as a C string with a project-unique symbol.
# Usage: cmake -DINPUT_FILE=<in> -DOUTPUT_FILE=<out.cpp> -DSYMBOL_NAME=<symbol> -P embed_text.cmake
file(READ ${INPUT_FILE} RL_TOOLS_EMBED_CONTENT)
if(RL_TOOLS_EMBED_CONTENT MATCHES "\\)RL_TOOLS_EMBED\"")
    message(FATAL_ERROR "embed_text.cmake: input contains the raw-string delimiter )RL_TOOLS_EMBED\"")
endif()
file(WRITE ${OUTPUT_FILE} "extern \"C\" const char ${SYMBOL_NAME}[] = R\"RL_TOOLS_EMBED(${RL_TOOLS_EMBED_CONTENT})RL_TOOLS_EMBED\";\n")
