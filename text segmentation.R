#https://medium.com/@SAPCAI/text-clustering-with-r-an-introduction-for-data-scientists-c406e7454e76
#https://github.com/vineetdhanawat/twitter-sentiment-analysis/blob/master/datasets/Sentiment%20Analysis%20Dataset.csv
#http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
# Loading the packages that will be used
list.of.packages <- c("tm", "dbscan", "proxy", "colorspace")
# (downloading and) requiring packages
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) 
  install.packages(new.packages)
for (p in list.of.packages) 
  require(p, character.only = TRUE)
rm(list = ls()) # Cleaning environment
#options(header = FALSE, stringsAsFactors = FALSE, fileEncoding = "latin1")

require(tm)


################################

#dataframe <- read.csv("C:\\Users\\justi\\Documents\\Text Segmentation IV\\Sentiment Analysis Dataset.csv.txt" )
dataframe <- read.csv("C:\\Users\\justi\\Documents\\Text Segmentation IV\\training.1600000.processed.noemoticon.csv")

unique(dataframe$Sentiment)
colnames(dataframe)<- c('polarity','tweet ID', 'date','query', 'username', 'TweetText')

sentences <- sub("http://([[:alnum:]|[:punct:]])+", '', dataframe$TweetText)
sentences <- sentences[1:1000]
corpus <- VCorpus(tm::VectorSource(sentences))



# Cleaning up
# Handling UTF-8 encoding problem from the dataset
corpus.cleaned <- tm::tm_map(corpus, function(x) iconv(x, to='UTF-8', sub='byte'))
corpus.cleaned <- tm_map(corpus, removePunctuation)
corpus.cleaned <- tm::tm_map(corpus.cleaned , content_transformer(tolower))
corpus.cleaned <- tm::tm_map(corpus.cleaned, tm::removeWords, tm::stopwords('SMART')) # Removing stop-words
corpus.cleaned <- tm::tm_map(corpus.cleaned, tm::stemDocument, language = "english") # Stemming the words 
corpus.cleaned <- tm::tm_map(corpus.cleaned, tm::stripWhitespace) # Trimming excessive whitespaces

corpus.cleaned[[1]]
corpus.cleaned[[2]]
corpus.cleaned[[3]]
corpus.cleaned[[4]]
corpus.cleaned[[5]]
corpus.cleaned[[6]]


# Building the feature matrices
tdm <- tm::DocumentTermMatrix(corpus.cleaned)

#rowTotals <- apply(tdm , 1, sum) #Find the sum of words in each Document
#tdm   <- tdm[rowTotals> 0, ]           #remove all docs without words

#tdm <- as.matrix(tdm)
#require(proxy)
#cosine_dist_mat <- as.matrix(dist(t(tdm), method = "cosine"))

tdm.tfidf <- tm::weightTfIdf(tdm)

# We remove A LOT of features. R is natively very weak with high dimensional matrix
tdm.tfidf <- tm::removeSparseTerms(tdm.tfidf, 0.999)
# There is the memory-problem part
# - Native matrix isn't "sparse-compliant" in the memory
# - Sparse implementations aren't necessary compatible with clustering algorithms
tfidf.matrix <- as.matrix(tdm.tfidf)
# Cosine distance matrix (useful for specific clustering algorithms)
dist.matrix = proxy::dist(tfidf.matrix, method = "cosine")
dist.matrix

dist.matrix = as.matrix(dist.matrix)
dims = dim(dist.matrix)

sink("C:\\Users\\justi\\Documents\\Text Segmentation IV\\t.gdf")
cat("nodedef>name VARCHAR,label VARCHAR\n")
for (a in 1:dims[1])
{  
  cat(paste(a,corpus.cleaned[[a]]$content,sep = ','))
  cat("\n")
}
cat('edgedef>node1 VARCHAR,node2 VARCHAR, weight DOUBLE\n')

for (a in 1:(dims[1]-1))
  for (b in (a+1):dims[1])
    {
      if (dist.matrix[a,b] < 0.999)
      {
        cat(paste(a, b, dist.matrix[a,b], sep=','))
        cat('\n')
      }
      #if (dist.matrix[a,b] < 0.5)
      #{
      #  print(corpus.cleaned[[a]]$content)
      #  print(corpus.cleaned[[b]]$content)
      #}
    }

sink()