# Generated by Django 3.0 on 2019-12-04 14:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('face_detection', '0002_auto_20191202_1840'),
    ]

    operations = [
        migrations.AddField(
            model_name='customer',
            name='last_visit',
            field=models.DateTimeField(auto_now=True, verbose_name='Date of last visit'),
        ),
        migrations.AddField(
            model_name='regularcustomer',
            name='last_visit',
            field=models.DateTimeField(auto_now=True, verbose_name='Date of last visit'),
        ),
    ]
