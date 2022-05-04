from transformers import BertModel, BertConfig, BertPreTrainedModel
import torch
import torch.nn as nn

class CustomBertForSequenceClassification(nn.Module):
    def __init__(self, config):
        super(CustomBertForSequenceClassification, self).__init__()
        self.num_labels=config.num_labels
        self.config=config
        self.bert=BertModel.from_pretrained("bert-base-uncased")
        classifier_dropout=(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, finetune=False, freeze=False, last=False):
        return_dict=return_dict if return_dict is not None else self.config.use_return_dict
        if freeze:
            with torch.no_grad():
                outputs=self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
                pooled_output=outputs[1]
                pooled_output=self.dropout(pooled_output)
        else:
            if finetune:
                outputs=self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
                pooled_output=outputs[1]
                pooled_output=self.dropout(pooled_output)
            else:
                with torch.no_grad():
                    outputs=self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
                    pooled_output=outputs[1]
                    pooled_output=self.dropout(pooled_output)
        logits=self.classifier(pooled_output)
        if last:
            return logits, pooled_output
        else:
            return logits
    
    def get_embedding_dim(self):
        return 768