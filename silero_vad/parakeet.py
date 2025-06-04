import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
asr_model.save_to("./my_local_model")


local_model = nemo_asr.models.ASRModel.restore_from("./my_local_model")


output = local_model.transcribe(['2086-149220-0033.wav'])
print(output[0])
# print(output[0].text)



# import nemo.collections.asr as nemo_asr

# # Load the pretrained model from HuggingFace Hub
# asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")

# # Save it locally using NeMo API
# asr_model.save_to("my_local_model.nemo")

# # Restore the model from the saved .nemo file
# local_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from("my_local_model.nemo")

# # Transcribe an audio file
# output = local_model.transcribe(['2086-149220-0033.wav'])
# print(output[0])
