library(SAVERX)

filepretrained <- saverx(
		"./hemato_data_raw.txt",
		data.species = "Human",
		is.large.data = TRUE,
		use.pretrain = TRUE,
		pretrained.weights.file = "SAVERX_pretrained_weights/human_Immune.hdf5",
		model.species = "Human")
print(paste0('The results are stored at: ', filepretrained))
