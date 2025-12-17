import torch
# print(torch._version_)        # Should show a version like 2.x.x.dev2025...
print(torch.version.cuda)       # Should print '12.9'
print(torch.cuda.is_available()) # Should be True