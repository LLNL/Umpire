.. _logging:

========================
Logging of Umpire Events            
========================

When debugging memory operation problems, it is sometimes helpful to enable
Umpire's logging facility.  The logging functionality is enabled for default
builds unless -DENABLE_LOGGING='Off' has been specified in which case it is
disabled.

If Umpire logging is enabled, it may be controlled by setting the
UMPIRE_LOG_LEVEL environment variable to '"Error"', '"Warning"', '"Info"',
or '"Debug"'.  The '"Debug"' value is the most verbose.

When UMPIRE_LOG_LEVEL has been set, events will be logged to the standard
output.

