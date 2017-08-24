from django import forms

class AttentionForm(forms.Form):
	input_mode = forms.IntegerField()
	context = forms.CharField(widget=forms.Textarea)
	content = forms.CharField(widget=forms.Textarea)
	created_at = forms.DateTimeField()
