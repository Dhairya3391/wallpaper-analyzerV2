# 🧪 Wallpaper Analyzer V2 — Testing Checklist

This checklist covers all major features, edge cases, and UI/UX flows to ensure the application is robust, user-friendly, and production-ready. Use this as a guide for manual and functional testing.

---

## 1. **Setup & Environment**

- [ ] Backend installs and starts without errors (Python, dependencies, Pillow-SIMD, open_clip_torch, etc.)
- [ ] Frontend installs and builds without errors (Node, pnpm/yarn/npm, Next.js)
- [ ] Database (SQLite) is created and writable
- [ ] Works on both Mac (MPS) and other platforms (CPU/GPU)

---

## 2. **Basic Functionality**

- [ ] User can enter/select a valid directory path
- [ ] User can start analysis and see progress/feedback
- [ ] Images are loaded and displayed in the masonry grid
- [ ] No images: UI shows appropriate warning
- [ ] Invalid directory: UI shows error
- [ ] Large directories (1000+ images) are handled gracefully

---

## 3. **Clustering & Analysis**

- [ ] User can select clustering algorithm (MiniBatchKMeans/DBSCAN) in settings
- [ ] User can set cluster count (auto or manual) for MiniBatchKMeans
- [ ] Clusters are created and displayed correctly
- [ ] Cluster sizes match expected image counts
- [ ] Cluster labels (CLIP tags) are shown in filter panel, image cards, and preview
- [ ] Cluster representative image is correct
- [ ] Changing clustering settings updates results as expected

---

## 4. **Aesthetic & Brightness Scoring**

- [ ] Each image displays an aesthetic score (0-1, as %)
- [ ] Each image displays a perceptual brightness score (LAB L channel, 0-100)
- [ ] Low-aesthetic images are marked ("Low Score")
- [ ] Brightness values are reasonable for dark/light images

---

## 5. **Duplicate Detection**

- [ ] Duplicate images are detected and marked
- [ ] User can filter to show only duplicates
- [ ] Duplicates are not included in clusters (if expected)

---

## 6. **UI/UX & Navigation**

- [ ] Filter panel allows filtering by cluster and duplicates
- [ ] Cluster badges show both number and label
- [ ] Image preview dialog shows all badges, filename, and actions
- [ ] User can download, favorite, and share images from preview
- [ ] Responsive design: works on desktop, tablet, mobile
- [ ] Tooltips/help text are present for advanced settings
- [ ] Loading states, error states, and empty states are clear

---

## 7. **Performance & Parallelism**

- [ ] Batch processing is fast (uses correct batch size for device)
- [ ] Parallelism (ThreadPoolExecutor) is effective (CPU usage scales)
- [ ] GPU/MPS acceleration is used if available (check logs)
- [ ] Caching works: repeated analysis is much faster

---

## 8. **Settings & Configurability**

- [ ] User can change similarity and aesthetic thresholds
- [ ] User can set image processing limit (e.g., 100, 1000, 5000)
- [ ] Recursive directory analysis works
- [ ] Skipping duplicates/aesthetics works as expected

---

## 9. **Error Handling & Edge Cases**

- [ ] Invalid/corrupt images are skipped with a warning, not a crash
- [ ] Permission errors (read/write) are handled gracefully
- [ ] Database errors (locked, full) are handled
- [ ] API errors are shown in the UI
- [ ] Long-running analysis does not freeze UI
- [ ] User can retry after error

---

## 10. **Legacy & Compatibility**

- [ ] Legacy HTML UI (index.html) still works for basic analysis
- [ ] Results are consistent between legacy and new UI (where applicable)

---

## 11. **Documentation & Help**

- [ ] README is up to date and accurate
- [ ] All new features are documented
- [ ] Setup instructions are clear for Mac, Windows, Linux
- [ ] All dependencies and optional optimizations are listed

---

## 12. **Final Polish**

- [ ] No unused code, debug prints, or commented-out blocks
- [ ] No TypeScript or lint errors in frontend
- [ ] No Python warnings/errors on backend
- [ ] All UI elements are visually consistent and accessible
- [ ] All features work as described in the documentation

---

**Tip:** Check logs (backend and browser console) for hidden errors or warnings during testing.

---

**If all boxes are checked, your app is ready for production!**
