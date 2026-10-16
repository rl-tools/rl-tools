# Remove when the pinned OWL exposes context-level register and primitive-type limits.
if(NOT IS_DIRECTORY "${OWL_SOURCE_DIR}")
    message(FATAL_ERROR "OWL compile-options patch requires OWL_SOURCE_DIR")
endif()

function(rl_tools_patch_owl relative before after)
    set(path "${OWL_SOURCE_DIR}/${relative}")
    file(READ "${path}" content)
    string(FIND "${content}" "${after}" patched)
    if(NOT patched EQUAL -1)
        return()
    endif()
    string(REPLACE "${before}" "" remainder "${content}")
    string(LENGTH "${content}" content_length)
    string(LENGTH "${remainder}" remainder_length)
    string(LENGTH "${before}" before_length)
    math(EXPR matched_length "${content_length} - ${remainder_length}")
    if(NOT matched_length EQUAL before_length)
        message(FATAL_ERROR "Unexpected OWL source in ${relative}; review the compile-options patch against the pinned revision")
    endif()
    string(REPLACE "${before}" "${after}" content "${content}")
    file(WRITE "${path}" "${content}")
endfunction()

rl_tools_patch_owl("include/owl/owl_host.h" [=[owlContextSetNumPayloadValues(OWLContext context,
                              size_t numRayPayloadValues);]=] [=[owlContextSetNumPayloadValues(OWLContext context,
                              size_t numRayPayloadValues);

OWL_API void
owlContextSetMaxRegisterCount(OWLContext context, size_t maxRegisterCount);]=])

rl_tools_patch_owl("owl/impl.cpp" [=[OWL_API void
owlContextSetNumPayloadValues(OWLContext _context, size_t numRayPayloadValues)
{
  LOG_API_CALL();
  checkGet(_context)->setNumPayloadValues(numRayPayloadValues);
}]=] [=[OWL_API void
owlContextSetNumPayloadValues(OWLContext _context, size_t numRayPayloadValues)
{
  LOG_API_CALL();
  checkGet(_context)->setNumPayloadValues(numRayPayloadValues);
}

OWL_API void
owlContextSetMaxRegisterCount(OWLContext _context, size_t maxRegisterCount)
{
  LOG_API_CALL();
  checkGet(_context)->setMaxRegisterCount(maxRegisterCount);
}]=])

rl_tools_patch_owl("owl/Context.h" [=[    void setNumPayloadValues(size_t numPayloadValues);]=] [=[    void setNumPayloadValues(size_t numPayloadValues);
    void setMaxRegisterCount(size_t maxRegisterCount);]=])
rl_tools_patch_owl("owl/Context.h" [=[    int numPayloadValues = 2;]=] [=[    int numPayloadValues = 2;
    int maxRegisterCount = 0;]=])

rl_tools_patch_owl("owl/Context.cpp" [=[  void Context::setNumPayloadValues(size_t numPayloadValues)
  {
    for (auto device : getDevices()) {
      assert("check programs have not been built"
             && device->allActivePrograms.empty());
    }
    this->numPayloadValues = (int)numPayloadValues;
  }]=] [=[  void Context::setNumPayloadValues(size_t numPayloadValues)
  {
    for (auto device : getDevices()) {
      assert("check programs have not been built"
             && device->allActivePrograms.empty());
    }
    this->numPayloadValues = (int)numPayloadValues;
  }

  void Context::setMaxRegisterCount(size_t maxRegisterCount)
  {
    for (auto device : getDevices()) {
      assert("check programs have not been built"
             && device->allActivePrograms.empty());
    }
    this->maxRegisterCount = (int)maxRegisterCount;
  }]=])

rl_tools_patch_owl("owl/DeviceContext.cpp" [=[  if (!debug) {
    moduleCompileOptions.maxRegisterCount  = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT ;]=] [=[  if (!debug) {
    moduleCompileOptions.maxRegisterCount  = parent->maxRegisterCount;]=])

rl_tools_patch_owl("include/owl/owl_host.h" [=[owlContextSetMaxRegisterCount(OWLContext context, size_t maxRegisterCount);]=] [=[owlContextSetMaxRegisterCount(OWLContext context, size_t maxRegisterCount);

OWL_API void
owlContextSetPrimitiveTypeFlags(OWLContext context, unsigned int primitiveTypeFlags);]=])

rl_tools_patch_owl("owl/impl.cpp" [=[OWL_API void
owlContextSetMaxRegisterCount(OWLContext _context, size_t maxRegisterCount)
{
  LOG_API_CALL();
  checkGet(_context)->setMaxRegisterCount(maxRegisterCount);
}]=] [=[OWL_API void
owlContextSetMaxRegisterCount(OWLContext _context, size_t maxRegisterCount)
{
  LOG_API_CALL();
  checkGet(_context)->setMaxRegisterCount(maxRegisterCount);
}

OWL_API void
owlContextSetPrimitiveTypeFlags(OWLContext _context, unsigned int primitiveTypeFlags)
{
  LOG_API_CALL();
  checkGet(_context)->setPrimitiveTypeFlags(primitiveTypeFlags);
}]=])

rl_tools_patch_owl("owl/Context.h" [=[    void setMaxRegisterCount(size_t maxRegisterCount);]=] [=[    void setMaxRegisterCount(size_t maxRegisterCount);
    void setPrimitiveTypeFlags(unsigned int primitiveTypeFlags);]=])
rl_tools_patch_owl("owl/Context.h" [=[    int maxRegisterCount = 0;]=] [=[    int maxRegisterCount = 0;
    unsigned int primitiveTypeFlags = 0;]=])

rl_tools_patch_owl("owl/Context.cpp" [=[  void Context::setMaxRegisterCount(size_t maxRegisterCount)
  {
    for (auto device : getDevices()) {
      assert("check programs have not been built"
             && device->allActivePrograms.empty());
    }
    this->maxRegisterCount = (int)maxRegisterCount;
  }]=] [=[  void Context::setMaxRegisterCount(size_t maxRegisterCount)
  {
    for (auto device : getDevices()) {
      assert("check programs have not been built"
             && device->allActivePrograms.empty());
    }
    this->maxRegisterCount = (int)maxRegisterCount;
  }

  void Context::setPrimitiveTypeFlags(unsigned int primitiveTypeFlags)
  {
    for (auto device : getDevices()) {
      assert("check programs have not been built"
             && device->allActivePrograms.empty());
    }
    this->primitiveTypeFlags = primitiveTypeFlags;
  }]=])

rl_tools_patch_owl("owl/DeviceContext.cpp" [=[#if OWL_CAN_DO_SPHERES
   pipelineCompileOptions.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
#endif]=] [=[#if OWL_CAN_DO_SPHERES
   pipelineCompileOptions.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
#endif
    if (parent->primitiveTypeFlags != 0) {
      pipelineCompileOptions.usesPrimitiveTypeFlags = parent->primitiveTypeFlags;
    }]=])
