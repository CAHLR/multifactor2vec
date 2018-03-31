# multifactor2vec
This tool is to train multifactor course2vec model using basic softmax loss. You can also just train course2vec.

SYNOPSIS
	python representation_presenter.py [options]
  
OPTIONS

INPUT AND OUTPUT:
	-i inputfile_name
      sampled training data pairs(target word, context word) matrix
      
	-o outputfile_name
      saved model
      
  -c coursefile
      course dictionary
  
  -f factorfile
      factor dictionary (e.g., major, instructor)
      
  -t validation loss file
      record validation loss after each epoch
      
PARAMETERS
  -e epoch
      number of training epochs, default 10
  
  -d vector dimension
      default 300
   
  -m whether train on batch or mini-batch
      0: train on batch, 1: useing mini-batch
  
  -b batch size if train on mini-batch
      default 4096
   
  -l learning rate
      default 1e-3
   
  -v whether using validation set to calculate validation loss
      0: False, 1: True
    
  -s train pure course2vec model or multifactor2vec model
      0: course2vec, 1: multifactor2vec
    
