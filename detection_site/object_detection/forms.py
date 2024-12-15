from django import forms
from .models import ImageFeed, UploadedImage


class ImageFeedForm(forms.ModelForm):
    class Meta:
        model = ImageFeed
        fields = ['image']
        widgets = {
            'image': forms.FileInput(attrs={'accept': 'image/*'}),
        }
        help_texts = {
            'image': 'Upload an image file.',
        }


class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ['image', 'model_choice']
