from vncorenlp import VnCoreNLP

rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.2.jar.1", annotators="wseg", max_heap_size='-Xmx500m')
print(rdrsegmenter.tokenize("thì cũng giống như ba má mình đã từng bị béo phì rồi bị bệnh này bệnh kia những người thân quen của mình mình biết bị"))
