detect faces
   ↓
is_good_face()         (light filter)
   ↓
embedding
   ↓
known match?

YES
   ↓
confirm identity

NO
   ↓
buffer embeddings
   ↓
centroid
   ↓
unknown search

Case A
   ↓
update centroid

Case B
   ↓
compute_face_quality
   ↓
select best frame
   ↓
store unknown image
