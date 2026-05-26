import os
import cv2
import torch
import numpy as np
import SimpleITK as sitk
from PIL import Image
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from torchvision import transforms
from reportlab.pdfgen import canvas
from .models import ScanResult
from .forms import ScanForm
from .ai_models import AttentionUNet, get_classifier
from django.utils.timezone import now
# ------------------------------
# LOAD MODELS
# ------------------------------
seg_model = AttentionUNet()
seg_model.load_state_dict(torch.load(
    os.path.join(settings.BASE_DIR,'detector/ml_models/pancreas_attention_unet.pth'),
    map_location='cpu'))
seg_model.eval()

clf_model = get_classifier()
clf_model.load_state_dict(torch.load(
    os.path.join(settings.BASE_DIR,'detector/ml_models/cancer_classifier_efficientnet.pth'),
    map_location='cpu'))
clf_model.eval()

seg_tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

clf_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(3),
    transforms.ToTensor()
])

def index(request):
    return render(request,'index.html')
def about(request):
    return render(request,'about.html')
def model(request):
    return render(request,'model.html')


# ------------------------------
def login_page(request):
    if request.method=="POST":
        u=request.POST['username']
        p=request.POST['password']
        user=authenticate(username=u,password=p)
        if user:
            login(request,user)
            return redirect('dashboard')
    return render(request,'login.html')



# =====================================================
# IMPORTS
# =====================================================

import cv2
import torch
import numpy as np
import SimpleITK as sitk

from PIL import Image

from django.conf import settings
from django.shortcuts import render
from django.utils.timezone import now

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .forms import ScanForm
from .models import ScanResult

# =====================================================
# DASHBOARD VIEW
# =====================================================

# =====================================================
# IMPORTS
# =====================================================

# =====================================================
# IMPORTS
# =====================================================

import os
import cv2
import torch
import numpy as np
import SimpleITK as sitk

from PIL import Image

from django.conf import settings
from django.shortcuts import render
from django.utils.timezone import now

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .forms import ScanForm
from .models import ScanResult

# =====================================================
# DASHBOARD
# =====================================================
@login_required
def dashboard(request):

    result = None
    confidence = None

    image_url = None
    mask_url = None
    heatmap_url = None

    # =====================================================
    # POST
    # =====================================================

    if request.method == "POST":

        form = ScanForm(
            request.POST,
            request.FILES
        )

        if form.is_valid():

            # =====================================================
            # SAVE FILE
            # =====================================================

            patient = form.cleaned_data['patient_name']

            file = request.FILES['file']

            path = os.path.join(
                settings.MEDIA_ROOT,
                file.name
            )

            with open(path, 'wb+') as f:

                for chunk in file.chunks():
                    f.write(chunk)

            # =====================================================
            # READ MRI
            # =====================================================

            img = sitk.ReadImage(path)

            arr = sitk.GetArrayFromImage(img)

            # Middle slice
            mid = arr.shape[0] // 2

            sl = arr[mid]

            # Normalize
            sl = cv2.normalize(
                sl,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

            # =====================================================
            # SAVE ORIGINAL MRI
            # =====================================================

            pil = Image.fromarray(sl)

            preview_path = os.path.join(
                settings.MEDIA_ROOT,
                'preview.png'
            )

            pil.save(preview_path)

            image_url = settings.MEDIA_URL + "preview.png"

            # =====================================================
            # SEGMENTATION
            # =====================================================

            x = seg_tf(pil).unsqueeze(0)

            with torch.no_grad():

                pred = seg_model(x)[0][0].numpy()

            # Binary
            pred = (
                pred > 0.5
            ).astype(np.uint8)

            # Resize
            pred = cv2.resize(
                pred,
                (
                    sl.shape[1],
                    sl.shape[0]
                ),
                interpolation=cv2.INTER_NEAREST
            )

            # =====================================================
            # PANCREAS CLEANUP
            # =====================================================

            kernel = np.ones((5,5), np.uint8)

            pred = cv2.morphologyEx(
                pred,
                cv2.MORPH_OPEN,
                kernel
            )

            pred = cv2.morphologyEx(
                pred,
                cv2.MORPH_CLOSE,
                kernel
            )

            # Largest pancreas only
            cnts, _ = cv2.findContours(
                pred,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            clean_pancreas = np.zeros_like(pred)

            if cnts:

                largest = max(
                    cnts,
                    key=cv2.contourArea
                )

                cv2.drawContours(
                    clean_pancreas,
                    [largest],
                    -1,
                    1,
                    -1
                )

            pred = clean_pancreas

            # =====================================================
            # ROI
            # =====================================================

            cnts, _ = cv2.findContours(
                pred,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if cnts:

                c = max(
                    cnts,
                    key=cv2.contourArea
                )

                x1, y1, w, h = cv2.boundingRect(c)

                roi = sl[
                    y1:y1+h,
                    x1:x1+w
                ]

            else:

                roi = sl

            roi_pil = Image.fromarray(
                roi
            ).convert("RGB")

            # =====================================================
            # CLASSIFICATION
            # =====================================================

            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            x2 = clf_tf(
                roi_pil
            ).unsqueeze(0).to(device)

            clf_model.to(device)

            with torch.no_grad():

                out = clf_model(x2)

                prob = torch.softmax(
                    out,
                    1
                )[0][1].item()

            confidence = round(
                prob * 100,
                2
            )

            if prob > 0.5:

                result = (
                    "Pancreatic Cancer Detected"
                )

            else:

                result = (
                    "No Cancer Detected"
                )

            # =====================================================
            # GRADCAM
            # =====================================================

            target_layers = [
                clf_model.features[-1]
            ]

            cam = GradCAM(
                model=clf_model,
                target_layers=target_layers
            )

            targets = [
                ClassifierOutputTarget(1)
            ]

            grayscale_cam = cam(
                input_tensor=x2,
                targets=targets
            )[0]

            # Resize CAM
            grayscale_cam = cv2.resize(
                grayscale_cam,
                (
                    sl.shape[1],
                    sl.shape[0]
                )
            )

            # =====================================================
            # TUMOR EXTRACTION
            # =====================================================

            # Normalize CAM
            grayscale_cam = cv2.normalize(
                grayscale_cam,
                None,
                0,
                1,
                cv2.NORM_MINMAX
            )

            # Strong cancer activations only
            tumor_region = (
                grayscale_cam > 0.70
            ).astype(np.uint8)

            # Only inside pancreas
            tumor_region = (
                tumor_region * pred
            )

            # =====================================================
            # REMOVE NOISE
            # =====================================================

            kernel = np.ones((3,3), np.uint8)

            tumor_region = cv2.morphologyEx(
                tumor_region,
                cv2.MORPH_OPEN,
                kernel
            )

            tumor_region = cv2.morphologyEx(
                tumor_region,
                cv2.MORPH_CLOSE,
                kernel
            )

            # =====================================================
            # KEEP ONLY INNER TUMOR
            # =====================================================

            cnts_tumor, _ = cv2.findContours(
                tumor_region,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            clean_tumor = np.zeros_like(
                tumor_region
            )

            for c in cnts_tumor:

                area = cv2.contourArea(c)

                # Small inner tumor only
                if 20 < area < 1200:

                    cv2.drawContours(
                        clean_tumor,
                        [c],
                        -1,
                        1,
                        -1
                    )

            tumor_region = clean_tumor

            # =====================================================
            # GROUND TRUTH MASK
            # =====================================================

            gt_mask_img = np.zeros(
                (
                    sl.shape[0],
                    sl.shape[1],
                    3
                ),
                dtype=np.uint8
            )

            # Green pancreas
            gt_mask_img[
                pred == 1
            ] = [0, 255, 0]

            # Red tumor
            gt_mask_img[
                tumor_region == 1
            ] = [0, 0, 255]

            # Save mask
            mask_path = os.path.join(
                settings.MEDIA_ROOT,
                'mask.png'
            )

            cv2.imwrite(
                mask_path,
                gt_mask_img
            )

            mask_url = (
                settings.MEDIA_URL +
                "mask.png"
            )

            # =====================================================
            # OVERLAY IMAGE
            # =====================================================

            overlay = cv2.cvtColor(
                sl,
                cv2.COLOR_GRAY2BGR
            )

            # Green pancreas contour
            cnts, _ = cv2.findContours(
                pred,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(
                overlay,
                cnts,
                -1,
                (0, 255, 0),
                2
            )

            # Light green pancreas
            green_mask = np.zeros_like(
                overlay
            )

            green_mask[
                pred == 1
            ] = [0, 255, 0]

            overlay = cv2.addWeighted(
                overlay,
                1.0,
                green_mask,
                0.25,
                0
            )

            # Red tumor
            red_mask = np.zeros_like(
                overlay
            )

            red_mask[
                tumor_region == 1
            ] = [0, 0, 255]

            final_overlay = cv2.addWeighted(
                overlay,
                1.0,
                red_mask,
                0.75,
                0
            )

            # =====================================================
            # SAVE OVERLAY
            # =====================================================

            heat_path = os.path.join(
                settings.MEDIA_ROOT,
                'heatmap.png'
            )

            cv2.imwrite(
                heat_path,
                final_overlay
            )

            heatmap_url = (
                settings.MEDIA_URL +
                "heatmap.png"
            )

            # =====================================================
            # SAVE DATABASE
            # =====================================================

            ScanResult.objects.create(
                patient_name=patient,
                file=file,
                result=result,
                confidence=confidence
            )

    else:

        form = ScanForm()

    # =====================================================
    # DASHBOARD STATS
    # =====================================================

    history = (
        ScanResult.objects
        .all()
        .order_by('-id')
    )

    total_scans = (
        ScanResult.objects.count()
    )

    detected_cases = (
        ScanResult.objects.filter(
            result="Pancreatic Cancer Detected"
        ).count()
    )

    today_reports = (
        ScanResult.objects.filter(
            created_at__date=now().date()
        ).count()
    )

    # =====================================================
    # RENDER
    # =====================================================

    return render(
        request,
        'dashboard.html',
        {

            'form': form,

            'result': result,
            'confidence': confidence,

            'image_url': image_url,
            'mask_url': mask_url,
            'heatmap_url': heatmap_url,

            'history': history,

            'total_scans': total_scans,
            'detected_cases': detected_cases,
            'today_reports': today_reports,
        }
    )

'''
def dashboard(request):

    result = None
    confidence = None

    image_url = None
    mask_url = None
    heatmap_url = None

    # =====================================================
    # POST REQUEST
    # =====================================================

    if request.method == "POST":

        form = ScanForm(request.POST, request.FILES)

        if form.is_valid():

            # =====================================================
            # SAVE FILE
            # =====================================================

            patient = form.cleaned_data['patient_name']

            file = request.FILES['file']

            path = os.path.join(
                settings.MEDIA_ROOT,
                file.name
            )

            with open(path, 'wb+') as f:

                for chunk in file.chunks():
                    f.write(chunk)

            # =====================================================
            # READ MRI (.MHA)
            # =====================================================

            img = sitk.ReadImage(path)

            arr = sitk.GetArrayFromImage(img)

            # Middle slice
            mid = arr.shape[0] // 2

            sl = arr[mid]

            # Normalize
            sl = cv2.normalize(
                sl,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

            # =====================================================
            # SAVE ORIGINAL MRI
            # =====================================================

            pil = Image.fromarray(sl)

            preview_path = os.path.join(
                settings.MEDIA_ROOT,
                'preview.png'
            )

            pil.save(preview_path)

            image_url = settings.MEDIA_URL + "preview.png"

            # =====================================================
            # SEGMENTATION
            # =====================================================

            x = seg_tf(pil).unsqueeze(0)

            with torch.no_grad():

                pred = seg_model(x)[0][0].numpy()

            # Binary mask
            pred = (pred > 0.5).astype(np.uint8)

            # Resize mask
            pred = cv2.resize(
                pred,
                (sl.shape[1], sl.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            # =====================================================
            # FIND ROI
            # =====================================================

            cnts, _ = cv2.findContours(
                pred,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if cnts:

                c = max(
                    cnts,
                    key=cv2.contourArea
                )

                x1, y1, w, h = cv2.boundingRect(c)

                roi = sl[y1:y1+h, x1:x1+w]

            else:

                roi = sl

            roi_pil = Image.fromarray(
                roi
            ).convert("RGB")

            # =====================================================
            # CLASSIFICATION
            # =====================================================

            x2 = clf_tf(roi_pil).unsqueeze(0)

            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            x2 = x2.to(device)

            clf_model.to(device)

            with torch.no_grad():

                out = clf_model(x2)

                prob = torch.softmax(
                    out,
                    1
                )[0][1].item()

            confidence = round(prob * 100, 2)

            if prob > 0.5:

                result = "Pancreatic Cancer Detected"

            else:

                result = "No Cancer Detected"

            # =====================================================
            # GRAD-CAM
            # =====================================================

            target_layers = [clf_model.features[-1]]

            cam = GradCAM(
                model=clf_model,
                target_layers=target_layers
            )

            targets = [ClassifierOutputTarget(1)]

            grayscale_cam = cam(
                input_tensor=x2,
                targets=targets
            )[0]

            # Resize GradCAM
            grayscale_cam = cv2.resize(
                grayscale_cam,
                (sl.shape[1], sl.shape[0])
            )

            # =====================================================
            # SHOW TUMOR ONLY INSIDE PANCREAS
            # =====================================================

            # Threshold
            tumor_region = (
                grayscale_cam > 0.60
            ).astype(np.uint8)

            # Keep only pancreas area
            tumor_region = tumor_region * pred

            # Remove noise
            kernel = np.ones((5,5), np.uint8)

            tumor_region = cv2.morphologyEx(
                tumor_region,
                cv2.MORPH_OPEN,
                kernel
            )

            # Keep largest tumor only
            cnts_tumor, _ = cv2.findContours(
                tumor_region,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            clean_tumor = np.zeros_like(
                tumor_region
            )

            if cnts_tumor:

                largest = max(
                    cnts_tumor,
                    key=cv2.contourArea
                )

                cv2.drawContours(
                    clean_tumor,
                    [largest],
                    -1,
                    1,
                    -1
                )

            tumor_region = clean_tumor

            # =====================================================
            # CREATE GROUND TRUTH MASK
            # =====================================================

            gt_mask_img = np.zeros(
                (
                    sl.shape[0],
                    sl.shape[1],
                    3
                ),
                dtype=np.uint8
            )

            # Green pancreas
            gt_mask_img[pred == 1] = [0, 255, 0]

            # Red tumor
            gt_mask_img[tumor_region == 1] = [0, 0, 255]

            # Save mask
            mask_path = os.path.join(
                settings.MEDIA_ROOT,
                'mask.png'
            )

            cv2.imwrite(
                mask_path,
                gt_mask_img
            )

            mask_url = settings.MEDIA_URL + "mask.png"

            # =====================================================
            # CREATE FINAL OVERLAY
            # =====================================================

            overlay = cv2.cvtColor(
                sl,
                cv2.COLOR_GRAY2BGR
            )

            # Green pancreas contour
            cv2.drawContours(
                overlay,
                cnts,
                -1,
                (0, 255, 0),
                2
            )

            # Light green pancreas
            green_mask = np.zeros_like(
                overlay
            )

            green_mask[pred == 1] = [0, 255, 0]

            overlay = cv2.addWeighted(
                overlay,
                1.0,
                green_mask,
                0.25,
                0
            )

            # Red tumor
            red_mask = np.zeros_like(
                overlay
            )

            red_mask[tumor_region == 1] = [0, 0, 255]

            final_overlay = cv2.addWeighted(
                overlay,
                1.0,
                red_mask,
                0.55,
                0
            )

            # =====================================================
            # SAVE FINAL OVERLAY
            # =====================================================

            heat_path = os.path.join(
                settings.MEDIA_ROOT,
                'heatmap.png'
            )

            cv2.imwrite(
                heat_path,
                final_overlay
            )

            heatmap_url = settings.MEDIA_URL + "heatmap.png"

            # =====================================================
            # SAVE DATABASE
            # =====================================================

            ScanResult.objects.create(
                patient_name=patient,
                file=file,
                result=result,
                confidence=confidence
            )

    else:

        form = ScanForm()

    # =====================================================
    # DASHBOARD DATA
    # =====================================================

    history = ScanResult.objects.all().order_by('-id')

    total_scans = ScanResult.objects.count()

    detected_cases = ScanResult.objects.filter(
        result="Pancreatic Cancer Detected"
    ).count()

    today_reports = ScanResult.objects.filter(
        created_at__date=now().date()
    ).count()

    # =====================================================
    # RENDER
    # =====================================================

    return render(request, 'dashboard.html', {

        'form': form,

        'result': result,
        'confidence': confidence,

        'image_url': image_url,
        'mask_url': mask_url,
        'heatmap_url': heatmap_url,

        'history': history,

        'total_scans': total_scans,
        'detected_cases': detected_cases,
        'today_reports': today_reports,
    })'''
'''
def dashboard(request):

    result = None
    confidence = None

    image_url = None
    mask_url = None
    heatmap_url = None

    # =====================================================
    # POST REQUEST
    # =====================================================

    if request.method == "POST":

        form = ScanForm(request.POST, request.FILES)

        if form.is_valid():

            # =====================================================
            # SAVE FILE
            # =====================================================

            patient = form.cleaned_data['patient_name']

            file = request.FILES['file']

            path = os.path.join(
                settings.MEDIA_ROOT,
                file.name
            )

            with open(path, 'wb+') as f:

                for chunk in file.chunks():
                    f.write(chunk)

            # =====================================================
            # READ MRI (.MHA)
            # =====================================================

            img = sitk.ReadImage(path)

            arr = sitk.GetArrayFromImage(img)

            # Middle slice
            mid = arr.shape[0] // 2

            sl = arr[mid]

            # Normalize
            sl = cv2.normalize(
                sl,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

            # =====================================================
            # SAVE ORIGINAL MRI
            # =====================================================

            pil = Image.fromarray(sl)

            preview_path = os.path.join(
                settings.MEDIA_ROOT,
                'preview.png'
            )

            pil.save(preview_path)

            image_url = settings.MEDIA_URL + "preview.png"

            # =====================================================
            # SEGMENTATION
            # =====================================================

            x = seg_tf(pil).unsqueeze(0)

            with torch.no_grad():

                pred = seg_model(x)[0][0].numpy()

            # Binary mask
            pred = (pred > 0.5).astype(np.uint8)

            # Resize mask
            pred = cv2.resize(
                pred,
                (sl.shape[1], sl.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            # =====================================================
            # FIND ROI
            # =====================================================

            cnts, _ = cv2.findContours(
                pred,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if cnts:

                c = max(
                    cnts,
                    key=cv2.contourArea
                )

                x1, y1, w, h = cv2.boundingRect(c)

                roi = sl[y1:y1+h, x1:x1+w]

            else:

                roi = sl

            roi_pil = Image.fromarray(
                roi
            ).convert("RGB")

            # =====================================================
            # CLASSIFICATION
            # =====================================================

            x2 = clf_tf(roi_pil).unsqueeze(0)

            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            x2 = x2.to(device)

            clf_model.to(device)

            with torch.no_grad():

                out = clf_model(x2)

                prob = torch.softmax(
                    out,
                    1
                )[0][1].item()

            confidence = round(prob * 100, 2)

            if prob > 0.5:

                result = "Pancreatic Cancer Detected"

            else:

                result = "No Cancer Detected"

            # =====================================================
            # GRAD-CAM
            # =====================================================

            target_layers = [clf_model.features[-1]]

            cam = GradCAM(
                model=clf_model,
                target_layers=target_layers
            )

            targets = [ClassifierOutputTarget(1)]

            grayscale_cam = cam(
                input_tensor=x2,
                targets=targets
            )[0]

            # =====================================================
            # RESIZE GRADCAM
            # =====================================================

            grayscale_cam = cv2.resize(
                grayscale_cam,
                (sl.shape[1], sl.shape[0])
            )

            # =====================================================
            # CREATE GROUND TRUTH MASK
            # =====================================================

            # Black image
            gt_mask_img = np.zeros(
                (sl.shape[0], sl.shape[1], 3),
                dtype=np.uint8
            )

            # -------------------------------------------------
            # GREEN = PANCREAS
            # -------------------------------------------------

            gt_mask_img[pred == 1] = [0, 255, 0]

            # -------------------------------------------------
            # RED = TUMOR REGION
            # -------------------------------------------------

            tumor_region = (
                grayscale_cam > 0.55
            ).astype(np.uint8)

            gt_mask_img[tumor_region == 1] = [0, 0, 255]

            # Save mask
            mask_path = os.path.join(
                settings.MEDIA_ROOT,
                'mask.png'
            )

            cv2.imwrite(
                mask_path,
                gt_mask_img
            )

            mask_url = settings.MEDIA_URL + "mask.png"

            # =====================================================
            # CREATE FINAL OVERLAY IMAGE
            # =====================================================

            overlay = cv2.cvtColor(
                sl,
                cv2.COLOR_GRAY2BGR
            )

            # -------------------------------------------------
            # PANCREAS CONTOUR (GREEN)
            # -------------------------------------------------

            cv2.drawContours(
                overlay,
                cnts,
                -1,
                (0, 255, 0),
                2
            )

            # Light green pancreas
            green_mask = np.zeros_like(overlay)

            green_mask[pred == 1] = [0, 255, 0]

            overlay = cv2.addWeighted(
                overlay,
                1.0,
                green_mask,
                0.25,
                0
            )

            # -------------------------------------------------
            # RED TUMOR REGION
            # -------------------------------------------------

            red_mask = np.zeros_like(overlay)

            red_mask[tumor_region == 1] = [0, 0, 255]

            final_overlay = cv2.addWeighted(
                overlay,
                1.0,
                red_mask,
                0.55,
                0
            )

            # =====================================================
            # SAVE OVERLAY IMAGE
            # =====================================================

            heat_path = os.path.join(
                settings.MEDIA_ROOT,
                'heatmap.png'
            )

            cv2.imwrite(
                heat_path,
                final_overlay
            )

            heatmap_url = settings.MEDIA_URL + "heatmap.png"

            # =====================================================
            # SAVE DATABASE
            # =====================================================

            ScanResult.objects.create(
                patient_name=patient,
                file=file,
                result=result,
                confidence=confidence
            )

    else:

        form = ScanForm()

    # =====================================================
    # DASHBOARD DATA
    # =====================================================

    history = ScanResult.objects.all().order_by('-id')

    total_scans = ScanResult.objects.count()

    detected_cases = ScanResult.objects.filter(
        result="Pancreatic Cancer Detected"
    ).count()

    today_reports = ScanResult.objects.filter(
        created_at__date=now().date()
    ).count()

    # =====================================================
    # RENDER
    # =====================================================

    return render(request, 'dashboard.html', {

        'form': form,

        'result': result,
        'confidence': confidence,

        'image_url': image_url,
        'mask_url': mask_url,
        'heatmap_url': heatmap_url,

        'history': history,

        'total_scans': total_scans,
        'detected_cases': detected_cases,
        'today_reports': today_reports,
    })'''
# ------------------------------
'''
@login_required
def dashboard(request):

    result=None
    confidence=None
    image_url=None
    mask_url=None
    heatmap_url=None

    if request.method == "POST":

        form = ScanForm(request.POST, request.FILES)

        if form.is_valid():

            # =====================================================
            # SAVE UPLOADED FILE
            # =====================================================

            patient = form.cleaned_data['patient_name']
            file = request.FILES['file']

            path = os.path.join(settings.MEDIA_ROOT, file.name)

            with open(path, 'wb+') as f:
                for chunk in file.chunks():
                    f.write(chunk)

            # =====================================================
            # READ MRI (.MHA)
            # =====================================================

            img = sitk.ReadImage(path)

            arr = sitk.GetArrayFromImage(img)

            # Take middle slice
            mid = arr.shape[0] // 2

            sl = arr[mid]

            # Normalize image
            sl = cv2.normalize(
                sl,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

            # Convert to PIL
            pil = Image.fromarray(sl)

            # Save preview image
            preview_path = os.path.join(
                settings.MEDIA_ROOT,
                'preview.png'
            )

            pil.save(preview_path)

            image_url = settings.MEDIA_URL + "preview.png"

            # =====================================================
            # SEGMENTATION
            # =====================================================

            x = seg_tf(pil).unsqueeze(0)

            with torch.no_grad():

                pred = seg_model(x)[0][0].numpy()

            # Binary Mask
            pred = (pred > 0.5).astype(np.uint8)

            # IMPORTANT:
            # Resize mask to original MRI size
            pred = cv2.resize(
                pred,
                (sl.shape[1], sl.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            # Save mask image
            mask_path = os.path.join(
                settings.MEDIA_ROOT,
                'mask.png'
            )

            cv2.imwrite(mask_path, pred * 255)

            mask_url = settings.MEDIA_URL + "mask.png"

            # =====================================================
            # FIND PANCREAS REGION
            # =====================================================

            cnts, _ = cv2.findContours(
                pred,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # ROI Crop
            if cnts:

                c = max(cnts, key=cv2.contourArea)

                x1, y1, w, h = cv2.boundingRect(c)

                roi = sl[y1:y1+h, x1:x1+w]

            else:

                roi = sl

            roi_pil = Image.fromarray(roi).convert("RGB")

            # =====================================================
            # CLASSIFICATION
            # =====================================================

            x2 = clf_tf(roi_pil).unsqueeze(0)

            with torch.no_grad():

                out = clf_model(x2)

                prob = torch.softmax(out, 1)[0][1].item()

            confidence = round(prob * 100, 2)

            if prob > 0.5:

                result = "Pancreatic Cancer Detected"

            else:

                result = "No Cancer Detected"

            # =====================================================
            # CREATE MEDICAL OVERLAY
            # Green = Pancreas
            # Red   = Cancer
            # =====================================================

            # Convert grayscale MRI to RGB
            overlay = cv2.cvtColor(
                sl,
                cv2.COLOR_GRAY2BGR
            )

            # Draw pancreas contour
            cv2.drawContours(
                overlay,
                cnts,
                -1,
                (0, 255, 0),   # Green
                2
            )

            # Create tumor mask
            tumor_mask = np.zeros_like(overlay)

            # Red cancer area
            tumor_mask[pred == 1] = [0, 0, 255]

            # Blend images
            final_overlay = cv2.addWeighted(
                overlay,
                1.0,
                tumor_mask,
                0.6,
                0
            )

            # =====================================================
            # LABELS
            # =====================================================

            cv2.putText(
                final_overlay,
                "Pancreas",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                final_overlay,
                "Cancer",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            # =====================================================
            # SAVE OVERLAY IMAGE
            # =====================================================

            heat_path = os.path.join(
                settings.MEDIA_ROOT,
                'heatmap.png'
            )

            cv2.imwrite(
                heat_path,
                final_overlay
            )

            heatmap_url = settings.MEDIA_URL + "heatmap.png"

            # =====================================================
            # SAVE DATABASE RESULT
            # =====================================================

            ScanResult.objects.create(
                patient_name=patient,
                file=file,
                result=result,
                confidence=confidence
            )

    else:

        form = ScanForm()

    # =====================================================
    # DASHBOARD STATS
    # =====================================================

    history = ScanResult.objects.all().order_by('-id')

    total_scans = ScanResult.objects.count()

    detected_cases = ScanResult.objects.filter(
        result="Pancreatic Cancer Detected"
    ).count()

    today_reports = ScanResult.objects.filter(
        created_at__date=now().date()
    ).count()

    # =====================================================
    # RENDER PAGE
    # =====================================================

    return render(request, 'dashboard.html', {

    'form': form,

    'result': result,
    'confidence': confidence,

    'image_url': image_url,
    'mask_url': mask_url,
    'heatmap_url': heatmap_url,

    'history': history,

    'total_scans': total_scans,
    'detected_cases': detected_cases,
    'today_reports': today_reports,
})
'''
# ------------------------------
@login_required
def report(request,id):
    row=ScanResult.objects.get(id=id)

    path=os.path.join(settings.MEDIA_ROOT,'report.pdf')

    c=canvas.Canvas(path)
    c.drawString(100,800,"Pancreas MRI Report")
    c.drawString(100,760,"Patient: "+row.patient_name)
    c.drawString(100,730,"Result: "+row.result)
    c.drawString(100,700,"Confidence: "+str(row.confidence)+"%")
    c.save()

    from django.http import FileResponse
    return FileResponse(open(path,'rb'),as_attachment=True)

def logout_page(request):
    logout(request)
    return redirect('login')