import re

substr = " ns"
#regex = re.compile(r'(\d+[.]\d+)'+substr)
#regex = re.compile(r'\d+?[.]?\d+$'+substr)
regex1 = re.compile(r'(\d+\d+)'+substr)
regex2 = re.compile("/"+r'(\d+\d+)')
lines = []
with open ('build/gcc-8-3-1_cuda-10-1_vendor-allocator-benchmarks-output_pt2.txt', 'rt') as myfile:
  for line in myfile:
    print(regex2.findall(line))
    print(regex1.findall(line))
    print('\n')
