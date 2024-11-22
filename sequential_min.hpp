int find_min(const int* ra, size_t size) {
    int ad_hoc_min = ra[0];
    for (size_t i = 1; i < size; ++i) {
        if (ra[i] < ad_hoc_min) {
            ad_hoc_min = ra[i];
        }
    }
    return ad_hoc_min;
}