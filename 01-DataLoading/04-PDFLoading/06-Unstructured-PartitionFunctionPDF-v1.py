from unstructured.partition.auto import partition
filename = "90-Data/black-myth-wukong/black-myth-wukong.pdf"
elements = partition(filename=filename,
                     content_type="application/pdf"
                    )
print("\n\n".join([str(el) for el in elements][:10]))
