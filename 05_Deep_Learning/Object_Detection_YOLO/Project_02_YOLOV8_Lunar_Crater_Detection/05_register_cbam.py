from ultralytics.nn import tasks

from attention.cbam import CBAM

# Register our custom module
tasks.CBAM = CBAM

print("CBAM registered successfully")