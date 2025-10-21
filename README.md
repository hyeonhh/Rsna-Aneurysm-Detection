# Rsna-Aneurysm-Detection : [분류 모델]
- https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/overview competition 실습 과정 기록을 위한 레포

### 데이터 구조 살펴보기
- `train.csv`
    - seriesInstanceUID
    - Modality : 이미지 종류(CTA, MRA, MRI)
    - PatientAge, PatientSex
    - 13개 뇌혈관 부위별 동맥류 여부 (0/1 , 부위별 동맥류 존재 여부)
    - Aneurysm Present : 전체 영상 내 동맥류 존재 여부(0/1)
- `train_localizers.csv` : 동맥류 위치 좌표를 담은 csv파일
    - SeriesInstanceUID : 스캔 시리즈 식별자
    - SOPInstanceUID : 시리즈 내 개별 이미지(슬라이스) 식별자
    - coordinates : 해당 슬라이스 위에서 동맥류 중심 xy좌표
    - Location : 동맥류 위치 설명
- `series/` : 훈련용 실제 DICOM 시리즈 데이터 저장 폴더
    - series/{seriesInstanceUID}/{SOPInstanceUID}.dcm
    - 각 폴더는 한 환자/시리지의 모든 슬라이스를 포함
- `segmentations/` : 일부 시리즈에 대한 혈관 분할 마스크(NifTI 파일, 3D) 폴더
    - 값은 각 혈관 부위의 고유 라벨 의미
    - 모델 학습/평가 시 관심영역 segmentation에 활용
- `kaggle_evaluation/` : Kaggle 제출 및 채점 환경에 필요한 평가 API용 파일 저장

### CT - Windowing(HU)
- Windowing, also known as grey-level mapping, contrast stretching, histogram modification or contrast enhancement is the process in which the CT image greyscale component of an image is manipulated via the CT numbers; doing this will change the appearance of the picture to highlight particular structures. The brightness of the image is adjusted via the window level. 
The contrast is adjusted via the window width.

- CT image의 그레이 스케일 성분을 CT number를 통해 조정하는 과정
- 특정 구조를 강조하도록 이미지를 변화시킨다.
- window level
    - 표시되는 CT값(HU)의 중심값
    - 윈도우 레벨을 낮추면 전체 이미지가 더 밝아지고, 높이면 더 어두워진다.
    - 이미지의 밝기를 조정
- window width
    - 표시되는 CT값(HU의 범위
    - 대조(조직 간의 차이)를 조정한다.
    - 윈도우 폭을 좁히면 작은 CT값 차이도 뚜렷하게 표현되어 조직 간 차이가 더 잘보인다.(대조 증가)
    - 윈도우 폭을 넓히면 더 많은 CT값이 한번에 표시되어 서로 다른 조직의 차이가 덜 두드러진다.(대조 감소)
 - https://radiopaedia.org/articles/windowing-ct?lang=us
   
### MRI - intensity normalization

- https://www.nature.com/articles/s41598-020-69298-z
- https://www.sciencedirect.com/science/article/pii/S0895611120300197
- https://arxiv.org/pdf/2406.01736

- MRI
  - <img width="1200" height="818" alt="image" src="https://github.com/user-attachments/assets/8d286c36-42f7-443e-bc52-f873ae4c23b5" />
  - large variability in image intensities in inter-patient and intra-patient comparisons
  - the intensity values of voxels do not have an absolute meaning unlike CT
  - can vary between scans
    - → CT는 HU 라는 단위가 있는데 MRI는 보셀 간 강도 값이 정해져 있지 않고 스캔 마다 다르다.
  - 그래서 MRI 이미지 전처리에서 image intensity를 줄여주는 작업이 중요하다
    - 이를 통해 다른 스캐너로 얻어진 이미지 간의 비교가 가능해진다.
  - 그러나 현재 MRI에서  정해진 정규화 기법이 없다 .
    - 관련 설명
            - MRI 영상의 **전처리(pre-processing) 기법**에 대한 설명
            - **Bias field correction**: MRI 영상에서 조직 내 강도 불균일(intensity inhomogeneity)을 줄이기 위한 보정 기법입니다. MRI는 자기장 불균일, 코일 특성 등으로 인해 동일 조직 내에서도 픽셀 값이 다르게 나타날 수 있는데, 이 현상을 보정해줍니다.
            - **Spatial resampling**: 서로 다른 voxel 크기(해상도)로 촬영된 MRI 데이터를 동일한 크기로 맞추는 과정입니다. 이렇게 하면 영상 간 비교나 분석이 쉬워집니다.[
            - **ROI definition**: 분석에 필요한 영역(예: 뇌 MRI에서 두개골을 제거하고 뇌만 남기는 brain extraction)을 미리 정의해서, 정규화나 분석을 해당 영역에만 적용하는 방법입니다. 이렇게 하면 불필요한 영역(두개골 등)이 분석에 영향을 주지 않게 할 수 있습니다.
            - **Normalization method**: MRI 영상의 정규화 방법(픽셀 값 분포를 일정하게 맞추는 방법)에 대해서는 radiomics 연구 분야에서 아직 합의가 없다는 점을 지적합니다. 즉, 여러 전처리 기법은 널리 쓰이지만, 어떤 정규화 방법이 가장 좋은지는 아직 논쟁 중이라는 의미입니다.
            
            ## 요약
            
            이 문단은 MRI 영상 분석에서 흔히 쓰이는 전처리 기법(강도 보정, 공간 리샘플링, ROI 정의 등)을 설명하면서, 정규화 방법에 대해서는 연구자들 사이에 합의가 없다는 점을 강조합니다. 즉, 영상의 품질과 분석 정확도를 높이기 위해 다양한 전처리 방법이 사용되지만, 정규화 방식은 연구마다 다를 수 있다는 뜻입니다.
            
- min-max normalization의 한계
  - <img width="1200" height="780" alt="image" src="https://github.com/user-attachments/assets/96b269c2-5c17-448c-8ff7-c46fb861b55d" />
  - <img width="1200" height="526" alt="image" src="https://github.com/user-attachments/assets/908aec56-954a-42ec-9a93-db44baab91b9" />

    - min-max 정규화는 이상치에 매우 민감하다.
    - **min-max 정규화**
      - 이미지의 최소값과 최대값을 기준으로 모든 픽셀 값을 0~1(혹은 0~255 등)로 스케일링합니다.
      - 만약 이미지에 극단적으로 높은 값이나 낮은 값(이상치)이 포함되어 있으면, 전체 정규화 범위가 그 이상치에 맞춰집니다.
      - 그 결과, 실제로 중요한 대부분의 픽셀 값들은 정규화 후 중간값 근처에 몰리게 되고, 이미지의 명암 대비(contrast)가 줄어들거나 정보가 압축(squeezed)되어 표현력이 떨어집니다.[arxiv+1](https://arxiv.org/pdf/2406.01736.pdf)
      - 즉, 이상치가 있으면 정규화된 이미지가 "평평해지고" 조직 간 차이가 잘 드러나지 않게 됩니다.
        
    -  ## 예시 
      - 예를 들어, MRI 이미지에 dead pixel이나 노이즈로 인해 아주 큰 값이 하나라도 있으면, min-max 정규화 후 대부분의 조직 신호가 0.4~0.6 같은 중간값에 몰리고, 실제 조직 간 차이가 잘 보이지 않게 됩니다.
      - 이런 문제 때문에, 1~99 분위수 정규화처럼 이상치를 잘라내고 정규화하는 방법이 더 효과적일 수 있습니다.[arxiv+1](https://arxiv.org/html/2405.08431v2)
- 1~99% normalization
  - <img width="1470" height="374" alt="image" src="https://github.com/user-attachments/assets/7db5dba7-b75a-4536-abc6-70fd36632781" /> 
  - **1~99 분위수 정규화**는 전체 픽셀 값 중에서 가장 낮은 1%와 가장 높은 1%를 잘라내고, 그 사이의 값만을 0~1(또는 0~255)로 정규화하는 방법입니다.
  - 이 방식은 min-max 정규화보다 이상치(outlier)에 덜 민감해서, 데이터에 따라 더 나은 선택이 될 수 있습니다.
  - 이 논문에서 다루는 radiomics 분석에서는 이 방법을 쓸 수 없었다고 합니다. 그 이유는, 분석 대상인 병변(lesion)의 픽셀 값이 전체 분포의 가장 높은 쪽(99% 이상)에 위치해 있었기 때문입니다.
  - 만약 1~99 분위수 정규화를 적용하면, 병변의 실제 신호(ROI 내부 값)까지 잘려나가거나(클리핑) 왜곡될 수 있습니다. 이렇게 되면 radiomics feature(조직 특성 추출)에 필요한 정보가 손실되어 분석에 부적합해집니다.
  - 1~99 분위수 정규화는 이상치가 많을 때 효과적이지만, 분석하고자 하는 조직(예: 병변)의 신호가 극단값에 위치한다면 오히려 중요한 정보를 잃을 수 있습니다.
    - 따라서, 정규화 방법을 선택할 때는 데이터의 분포와 분석 목적(특히 ROI의 위치와 특성)을 반드시 고려해야 한다는 점을 강조하는 내용입니다.
- z-score normalization
    - 이미지 값 - 평균 / 표준 편차
    - 한계
        - <img width="1200" height="180" alt="image" src="https://github.com/user-attachments/assets/71acd04b-4f70-4df5-9d07-1fe4fcc6437e" />
        - <img width="1200" height="866" alt="image" src="https://github.com/user-attachments/assets/f38f0bf7-fe34-4081-87b1-08f128351f1f" />
        - <img width="1200" height="252" alt="image" src="https://github.com/user-attachments/assets/0d8e0aaa-4aca-4dbd-b5cc-3da8b860cef8" />
        - <img width="1200" height="1008" alt="image" src="https://github.com/user-attachments/assets/f0ea7be7-b9dd-4111-82e9-b1ecbb4ac016" />

### 각 시리즈별 3D볼륨 구성하기

1. Xyz -> IRC
  - `train_localizers.csv`에서는 x,y좌표만 있는데  z를 어떻게 처리하지?
  - **동일한 series에서 같은 위치이면 다른 슬라이스에도 유사한 위치로 이어지는 건가?? **   
    - x,y는 슬라이스(2D 영상)위의 위치 ,z는 해당 슬라이스의 인덱스(3D 내 위치)
        - z값을 찾는 방법
            - SOPInstanceUID로 매칭되는 DICOM 파일을 읽는다.
            - 해당 DICOm의 파일리스트 또는 메타데이터에서
            - 배열 내 몇 번째 슬라이스(z인덱스)인지를 결정한다.
        
    - 다른 사람 코드 살펴봄
        - https://www.kaggle.com/code/sacuscreed/rsna2025-32ch-img-infer-lb-0-69-share
    
    ```python
    def extract_slice_info(self, datasets: List[pydicom.Dataset]) -> List[Dict]:
            """
            # 시리즈의 이미지 슬라이스 각각 에 대해서 
    				Extract position information for each slice
            """
           
            slice_info = []
            
            for i, ds in enumerate(datasets):
                info = {
                    'dataset': ds,
                    'index': i,
                    'instance_number': getattr(ds, 'InstanceNumber', i),
                }
                
                # Get z-coordinate from ImagePositionPatient and ImageOrientationPatient            
                try:
                    ipp = np.array(getattr(ds, 'ImagePositionPatient', None))
                    iop = np.array(getattr(ds, 'ImageOrientationPatient', None))
                    n_vec = np.cross(iop[:3],iop[3:])
    	         # np. cross : 두 벡터가 이루는 평면에 수직인 벡터를 반환한다. 
    	         # Dicom의 ImageOrientationPatient는 6개의 숫자로 구성되어있음
    	         # iop[:3]은 이미지의 행 방향을 나타내는 벡터
    	         # iop[3:]은 이미지의 열 방향을 나타내는 벡터
    	         # n_vec : 슬라이스 평면에 수직인 벡터 
                  
                    '''
    ## float((ipp*n_vec).sum())으로   z- position을 구함 
    - ImagePositionPatient : 해당 슬라이스의 왼쪽 위 모서리의 환자 기준 3D좌표를 나타냄, 슬라이스의 시작점 좌표 
    - ImageOrientationPatient : 이미지 방향 정보 -슬라이스의 행과 열 방향을 3D 벡터로 알려준다.
    - z 위치 계산하는 이유 : 여러 슬라이스를 올바른 순서로 쌓으려면 각 슬라이스가 환자 몸의 어디에 위치하는지 알아야한다.     
    - 시작점 좌표를 법선 벡터에 투영하면 슬라이스가 z축 상에서 어디에 있는지 알 수 있다.
    
                    '''
                    
                    info['z_position'] = -f오loat((ipp*n_vec).sum())
                except Exception as e:
                    info['z_position'] = float(i)
                    #print(f"Failed to extract position info: {e}")
                
                slice_info.append(info)
            
            return slice_info
    ```
    
    <aside>
    🐣
    
    - 그리고 CT 데이터가 아니라 MRA 도 있는데 이 경우도 IRC로 동일하게 변환하면 되나?
    </aside>
    
    - CT/MRA/CTA 모두 배열 인덱스(IRA)변환 방식은 동일하게 적용!
    - DICOM/nii 영상 모두 (Origin, Spacing, Direction) 정보를 기록하므로 환자 좌표계 X,Y,Z ↔ 배열 인덱스 계 (I,R,C) 변환을 똑같이 계산한다.
    
    <aside>
    🐣
    
    CT/CTA/MRA는 각각 뭐야?
    
    </aside>
    
    - CT : X선을 이용해서 인체를 단층으로 촬용
    - CTA : CT에 조영제를 주입하여 혈관을 선명하게 활용
    - MRA : MRI 기술을 사용해서 혈관만 선택적으로 영상화
    
    <aside>
    🐣
    
    train_localizers에는 동맥류의 x,y좌표가 있는데
    IRC로 변환하려면 어떤 걸 변환해야해?  ImagePositionPatient야? 아니면 동맥류의 x,y좌표?
    
    </aside>

### kaggle - predict 메소드 흐름 이해하기

- 예제 코드
    
    ```python
    ID_COL = 'SeriesInstanceUID'
    
    LABEL_COLS = [
    ]
    
    # All tags (other than PixelData and SeriesInstanceUID) that may be in a test set dcm file
    DICOM_TAG_ALLOWLIST = [
        'BitsAllocated',
        'BitsStored',
        'Columns',
        'FrameOfReferenceUID',
        'HighBit',
        'ImageOrientationPatient',
        'ImagePositionPatient',
        'InstanceNumber',
        'Modality',
        'PatientID',
        'PhotometricInterpretation',
        'PixelRepresentation',
        'PixelSpacing',
        'PlanarConfiguration',
        'RescaleIntercept',
        'RescaleSlope',
        'RescaleType',
        'Rows',
        'SOPClassUID',
        'SOPInstanceUID',
        'SamplesPerPixel',
        'SliceThickness',
        'SpacingBetweenSlices',
        'StudyInstanceUID',
        'TransferSyntaxUID',
    ]
    
    # Replace this function with your inference code.
    # You can return either a Pandas or Polars dataframe, though Polars is recommended.
    # Each prediction (except the very first) must be returned within 30 minutes of the series being provided.
    def predict(series_path: str) -> pl.DataFrame | pd.DataFrame:
        """Make a prediction."""
        # --------- Replace this section with your own prediction code ---------
        series_id = os.path.basename(series_path)
        
        all_filepaths = []
        for root, _, files in os.walk(series_path):
            for file in files:
                if file.endswith('.dcm'):
                    all_filepaths.append(os.path.join(root, file))
        all_filepaths.sort()
        
        # Collect tags from the dicoms
        tags = defaultdict(list)
        tags['SeriesInstanceUID'] = series_id
        global dcms
        for filepath in all_filepaths:
            ds = pydicom.dcmread(filepath, force=True)
            tags['filepath'].append(filepath)
            for tag in DICOM_TAG_ALLOWLIST:
                tags[tag].append(getattr(ds, tag, None))
            # The image is in ds.PixelData
    
        # ... do some machine learning magic ...
        
        '''
        여기서 모델에 대해 예측한 predictions을 지정한 형태에 맞춰서 넘겨줘야한다. 
        '''
        predictions = pl.DataFrame(
            data=[[series_id] + [0.5] * len(LABEL_COLS)],
            schema=[ID_COL, *LABEL_COLS],
            orient='row',
        )
        # ----------------------------------------------------------------------
    
        if isinstance(predictions, pl.DataFrame):
            assert predictions.columns == [ID_COL, *LABEL_COLS]
        elif isinstance(predictions, pd.DataFrame):
            assert (predictions.columns == [ID_COL, *LABEL_COLS]).all()
        else:
            raise TypeError('The predict function must return a DataFrame')
    
        # ----------------------------- IMPORTANT ------------------------------
        # You MUST have the following code in your `predict` function
        # to prevent "out of disk space" errors. This is a temporary workaround
        # as we implement improvements to our evaluation system.
        shutil.rmtree('/kaggle/shared', ignore_errors=True)
        # ----------------------------------------------------------------------
        
        return predictions.drop(ID_COL)
    ```
    
  
    ```python
     # make ensemble prediction
            # 앙상블 예측 및 좌우 뒤집기 평균 
            # np.flip(volume, -2) -> 두번째 축(보통 width, 좌우)을 뒤집음 
            # 두 결과를 평균내어서 최종 예측으로 사용)
            final_pred = (predict_ensemble(volume) + predict_ensemble(np.flip(volume, -2))[[1,0,3,2,5,4,6,8,7,10,9,11,12,13]])/2
    
            # Create output dataframe
            prediction_df = pl.DataFrame(
                data = [ [series_id] + final_pred.tolist()], 
                schema = [ID_COL] + LABEL_COLS,
                orient = 'row'
            )
    
    ```
    
    ```python
      '''
      predict_ensemble(volume) 내부...
      각 폴드별 예측을 모으고, 이 예측들의 평균을 낸다. 
      
      '''
      
      
      predictions= np.array(all_predictions) # 폴드별 에측값을 numpy 배열로 변환 
    
        return np.average(predictions, weights = weights, axis = 0) # 폴드별 예측값을 가중 평가하여 최종 예측값을 반환 
    ```
  ----

  ### 데이터 훈련시켜보기 
  - 대용량 원본 데이터 중 필요한 데이터만 찾아볼 수 있도록 걸러낼 방법을 생각해보자
    
## 모델 구성해보기
- 이전에는 다른 사람이 만든 모델을 통해 코드가 작동하는 방식, 입력 데이터와 모델을 연결하는 방식을 익혔다면, 이제 성능이 나쁘더라고 모델을 직접 구현해보려고 한다.

- <img width="716" height="549" alt="image" src="https://github.com/user-attachments/assets/986b5709-74b6-4c8f-85bd-d34559d55689" />
- (출처 : Deeplearning with pytorch)

### 먼저 Dataset과 DataLoader 코드 작성
- Dataset & DataLoader 코드 짜보기
    ```python
    class RSNADataset(Dataset):
    def __init__(self,df):
        self.df = df
        self.series_uids = df['SeriesInstanceUID']
        
    def __len__(self):
        return len(self.series_uids)
    def __getitem__(self,ndx):
        try:
            series_uid = self.series_uids[ndx]
            path = os.path.join(SERIES_PATH, series_uid)
            volume = process_dicom_series_safe(path)
            # label도 같이 넘기기
            label = self.df[ndx]['Aneurysm Present']
            
        except Excepting as e:
            print(f"RSNA dataset : Error occurs while get item : {series_uid}")
            volume = None
            label = None
        return volume,label
    ```

    ```python
    ## DataLoader
    train_ds = RSNADataset(train_copy)
    train_loader = torch.utils.data.DataLoader(train_ds,batch_size = 10, shuffle= True)
    ```

#### series_uid별로 폴더가 구성되어있고, 내부에 dicom 이미지 파일이 있는데 series_uid로 반복문을 작성해야할까?
   - RSNADataset에서 DataLoader를 연결하면,  DataLoader가 series_uid들을 자동으로 호출해서 데이터를 불러온다.
   - RSNADataset의 `__len__`과 `__getitem__`을 구현했으므로 `DataLoader(train_ds)`를 만들면 내부적으로 `for idx in range(len(train_ds)):` 처럼 인덱스를 자동으로 생성해서 `trian_ds.__getitem__(ds)`를 반복적으로 호출해준다.
   - 결론적으로, `series_uid`를 직접 넘겨주거나 반복문을 작성할 필요 업이 `DataLoader`가 알아서 각 인덱스에 해당하는 `series_uid`를 가져와 DICOM 볼륨을 반환하는 구조 !! 
  
#### 모델 입출력 데이터 사이즈 
- 각 series_uid별 volume shape : `(32, 384, 384)`
- input_t.shape : `torch.Size([4, 1, 32, 384, 384])`
    - 배치 크기 4
    - 채널 크기 1
    - 깊이 32, 높이 384, 너비 384 
- input_Batch.shape :` torch.Size([2, 1, 32, 384, 384])`
    - 배치 크기 2
- block_out.shape : `torch.Size([2, 64, 2, 24, 24])`
    - 배치 크기 2
    - 채널 수 64로 증가
    - 깊이 2, 높이 24, 너비 24 
- conv_flat.shape : torch.Size([2, 73728])
    - block_out을 2D 텐서로 평탄화 64 * 2 * 24 * 2
    - 배치 크기 2,
####  모델 입출력 형태
- <img width="661" height="217" alt="image" src="https://github.com/user-attachments/assets/9135034d-cfef-43a1-8e75-55fa5783a574" />
- 클래스 인덱스(class indices)를 정답으로 넘기는 경우 target의 데이터 타입은 반드시 long
    - 예를 들어 [0][2][1] 처럼 각 원소가 정수형(long)이어야한다.
- 클래스 확률을 정답으로 넘기는 경우
    - target의 데이터타입은 반드시 float이어야하고 값은 0과 1사이이다.
    - 예 `[0., 1.]`, ` [0.2, 0.8]  `

```python
# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5) #.long을 해준다.
output = loss(input, target)
output.backward()

# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
output = loss(input, target)
output.backward()
```

#### 만들어진 volume 저장하기
- 매번 dicom 이미지를 volume으로 만드는 작업이 너 ~~ 무 오래 걸린다.
- 그래서 방법을 찾아보다가 데이터셋으로 저장할 수 있다는 걸 알게되었다!
- series_uid에 맞는 dicom 이미지가 만들어지면 저장하도록 했다.
- 이후 코드 실행 시 저장된 파일이 없는 경우에만 전처리를 하고, 있는 경우 해당 volume을 불러와서 쓰도록 한다.
- <img width="349" height="221" alt="image" src="https://github.com/user-attachments/assets/95a3f714-183a-4191-9c37-fa1d297a1e26" />

