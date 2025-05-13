# Enter python code
import sys
sys.path.append("/home/dataguess/notebooks/berkan/sort")
from sort import *
import cv2
import numpy as np
from time import time

threshold = 0.35
classes = ['insan']

colors = [			# Kırmızı (Red)
		(0, 255, 0),			# Yeşil (Green)
		(255, 0, 0),			# Mavi (Blue)
		(255, 0, 255),		# Mor (Magenta)
		(200, 95, 185),		# Turkuaz (Cyan)
		(0, 128, 128),		# Koyu Kırmızı (Dark Red)
		(0, 128, 128),		# Zeytin Yeşili (Olive)
		(0, 128, 0),			# Koyu Yeşil (Dark Green)
		(128, 0, 128),		# Koyu Mor (Dark Magenta)
		(128, 0, 0),			# Koyu Turkuaz (Dark Cyan)
		(128, 0, 0),			# Koyu Mavi (Dark Blue)
		(128, 128, 128),	# Orta Gri (Medium Gray)
		(0, 165, 255),		# Turuncu (Orange)
		(147, 20, 255),	 # Derin Pembe (Deep Pink)
		(127, 255, 0),		# Yüksek Yeşil (Medium Spring Green)
		(19, 69, 139),		# Kahverengi (Brown)
		(180, 105, 255),	# Sıcak Pembe (Hot Pink)
]

#s
rois = [
		[(330,260),(420,260),(420,360),(330,360)],
		[(220,200),(310,200),(310,260),(220,260)],
		[(240,70),(315,70),(315,120),(240,120)],
		[(360,30),(430,30),(430,80),(360,80)],
		[(540,65),(630,65),(630,115),(540,115)],
		[(670,135),(780,135),(780,195),(670,195)]
]
#s

def getColorbyId(color_id):
		global colors

		if 0 <= color_id < len(colors):
				return colors[color_id]

counter=0
#.... Initialize SORT .... 
#......................... 
sort_max_age = 75 # def 1
sort_min_hits = 2 # def 3
sort_iou_thresh = 0.3 # 0.3
sort_tracker = Sort(max_age=sort_max_age,
								min_hits=sort_min_hits,
								iou_threshold=sort_iou_thresh) 



def plot(img, box, label, clsid, color=(0,0,255), font=cv2.FONT_HERSHEY_SIMPLEX, thickness=2):
	
	color=getColorbyId(clsid)
	# circle
	box = np.array(box, dtype=float).astype(int)
	center = box[:2] + ((box[2:] - box[:2]) / 2)
	img = cv2.circle(img, center=center.astype(np.int32), radius=2, color=color, thickness=thickness)
	
	# custom edges
	x1, y1, x2, y2 = box
	r, d = 10, 10
	

	#img = cv2.rectangle(img, (x1,y1), (x2,y2), color, 1)
	# label
	(w, h), _ = cv2.getTextSize(label, font, 0.65, thickness)
# 	img = cv2.rectangle(img, (x2-25, y2 - 15), (x2 + w -20 , y2 + 10), color, -1)
# 	img = cv2.putText(img, label, (x2-20, y2 + 5), font, 0.65, (255,255,255), thickness)
	#drawRoi(img)
	
	return img	

# ROI'ler için süre bilgilerini tutacak sözlük
roi_timers = {i: {"objects": {}} for i in range(len(rois))}

def run(getInputValueById, setOutputValueById):
		global counter, roi_timers
		model_out = getInputValueById('f17d0737-a14a-4090-a263-cb56ee0b6333')
		ratio = getInputValueById('ba2dc919-10bf-4d40-a814-67bb17b0f3f4')
		dwdh = getInputValueById('f23bc13d-63b6-4f48-b1e8-47a2628d517d')
		img = getInputValueById('81958325-65bc-4d36-8521-2877bc7e6f68')
		img_cpy = img.copy()

		dets_to_sort = np.empty((0, 5))
		class_to_sort = []
		
		# ROI'ler için flag ve süre bilgilerini güncelle
		roi_flag = [0] * len(rois)
		
		for _, x0, y0, x1, y1, cls_id, score in model_out:
				if score < threshold:
						continue
				box = np.array([x0, y0, x1, y1])
				box -= np.array(dwdh*2)
				box /= ratio
				box = box.round().astype(np.int32).tolist()
				
				center = (int(box[0] + (box[2]-box[0])/2), int(box[1] + (box[3]-box[1])/2))
				dets_to_sort = np.vstack((dets_to_sort, np.hstack((box, score))))
				class_to_sort.append(cls_id)		

		tracked_dets = sort_tracker.update(dets_to_sort, class_to_sort)
		current_time = time()

		for dets in tracked_dets:	# x0, y0, x1, y1, _id, _it, _score, _cls_id
				box = dets[:4].astype(int)
				labelid = round(dets[4])
				label = classes[int(dets[7])]
				score = int(dets[6] * 100)
				center = (int(box[0] + (box[2]-box[0])/2), int(box[1] + (box[3]-box[1])/2))

				# Her bir ROI için kontrol yap
				for i, roi in enumerate(rois):
						pts = np.array(roi, np.int32)
						pts = pts.reshape((-1, 1, 2))
								
						x_values = [p[0] for p in roi]
						y_values = [p[1] for p in roi]
								
						x_min, x_max = min(x_values), max(x_values)
						y_min, y_max = min(y_values), max(y_values)
		
						if x_min <= center[0] <= x_max and y_min <= center[1] <= y_max:
								roi_flag[i] = 1
								# Nesne ROI içinde
								if labelid not in roi_timers[i]["objects"]:
										# Nesne ilk kez ROI'ye girdi
										roi_timers[i]["objects"][labelid] = {"entry_time": current_time, "exit_time": None, "confirmed": False}
								else:
										# Nesne hala ROI içinde
										if roi_timers[i]["objects"][labelid]["exit_time"] is not None:
												# Nesne tekrar ROI'ye girdi
												if current_time - roi_timers[i]["objects"][labelid]["exit_time"] <= 5:
														# 5 saniye içinde geri döndü, süre devam eder
														roi_timers[i]["objects"][labelid]["exit_time"] = None
												else:
														# 5 saniyeden fazla oldu, yeni bir giriş olarak kabul et
														roi_timers[i]["objects"][labelid]["entry_time"] = current_time
														roi_timers[i]["objects"][labelid]["exit_time"] = None
														roi_timers[i]["objects"][labelid]["confirmed"] = False
										else:
												# Nesne hala ROI içinde, 5 saniye doldu mu kontrol et
												if not roi_timers[i]["objects"][labelid]["confirmed"] and (current_time - roi_timers[i]["objects"][labelid]["entry_time"]) >= 5:
														roi_timers[i]["objects"][labelid]["confirmed"] = True
						else:
								# Nesne ROI dışında
								if labelid in roi_timers[i]["objects"] and roi_timers[i]["objects"][labelid]["exit_time"] is None:
										# Nesne ilk kez ROI'den çıktı
										roi_timers[i]["objects"][labelid]["exit_time"] = current_time

				img_cpy = plot(img_cpy, box, f'{label}', int(dets[7]))
				
		# ROI'lerin sürelerini güncelle ve ekrana yaz
		for i, roi in enumerate(rois):
				pts = np.array(roi, np.int32)
				pts = pts.reshape((-1, 1, 2))

				# Süre sayımının devam edip etmediğini kontrol et
				is_timer_active = False
				for obj_id, obj_data in roi_timers[i]["objects"].items():
						if obj_data["confirmed"] and (obj_data["exit_time"] is None or (current_time - obj_data["exit_time"]) <= 3):
								is_timer_active = True
								break

				# ROI rengini belirle
				color_2 = (0, 255, 0) if is_timer_active else (0, 0, 255)	# Yeşil veya Kırmızı
				cv2.polylines(img_cpy, [pts], isClosed=True, color=color_2, thickness=2)

				# ROI'nin süresini hesapla
				total_duration = 0
				for obj_id, obj_data in roi_timers[i]["objects"].items():
						if obj_data["confirmed"]:	# Yalnızca 3 saniye sonrasını say
								entry_time = obj_data["entry_time"]
								exit_time = obj_data["exit_time"] if obj_data["exit_time"] is not None else current_time

								# Eğer ROI içinde başka nesneler varsa, süre sayımını kesme
								if any(other_obj["exit_time"] is None for other_id, other_obj in roi_timers[i]["objects"].items() if other_id != obj_id):
										total_duration += (current_time - entry_time)
								else:
										total_duration += (exit_time - entry_time)

				# Süreyi ROI'nin üzerine yaz
				#text = f"Time: {int(total_duration)}s"
				#cv2.putText(img_cpy, text, (roi[0][0], roi[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
				
		setOutputValueById('42fad565-162e-4200-932a-0835feedf35d', img_cpy)