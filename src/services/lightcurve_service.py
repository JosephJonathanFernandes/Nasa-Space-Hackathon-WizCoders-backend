import io
import logging
import matplotlib.pyplot as plt
import numpy as np
from fastapi import HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
import lightkurve as lk
import tempfile
from lightkurve import read
from astropy.io import fits as _fits

class LightCurveService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def get_lightcurve(self, kepler_id: str, mission: str = "Kepler"):
        try:
            self.logger.info(f"Searching for light curve: {kepler_id}, mission: {mission}")
            search_result = lk.search_lightcurve(kepler_id, mission=mission)
            if len(search_result) == 0:
                raise HTTPException(status_code=404, detail=f"No light curves found for {kepler_id} in {mission} mission")
            lc = search_result.download()
            if lc is None:
                raise HTTPException(status_code=404, detail=f"Could not download light curve for {kepler_id}")
            self.logger.info(f"Downloaded light curve with {len(lc)} data points")
            plt.figure(figsize=(12, 6))
            lc.plot()
            plt.title(f"{mission} Light Curve for {kepler_id}")
            plt.xlabel("Time")
            plt.ylabel("Flux")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            return StreamingResponse(io.BytesIO(img_buffer.read()), media_type="image/png", headers={"Content-Disposition": f"inline; filename={kepler_id}_lightcurve.png"})
        except Exception as e:
            self.logger.error(f"Error processing {kepler_id}: {str(e)}")
            plt.close('all')
            if isinstance(e, HTTPException):
                raise e
            else:
                raise HTTPException(status_code=500, detail=f"Internal server error while processing {kepler_id}: {str(e)}")

    async def upload_fits_file(self, file: UploadFile = File(...)):
        try:
            contents = await file.read()
            tmp_path = None
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
                tmp_path = tmp.name
                tmp.write(contents)
                tmp.flush()
                tmp.close()
                try:
                    lc = read(tmp_path)
                except Exception as _ll_err:
                    hdul = _fits.open(tmp_path, memmap=False)
                    time_arr = None
                    flux_arr = None
                    for hdu in hdul:
                        data = getattr(hdu, 'data', None)
                        if data is None:
                            continue
                        cols = []
                        try:
                            cols = list(data.columns.names)
                        except Exception:
                            try:
                                cols = list(data.dtype.names or [])
                            except Exception:
                                cols = []
                        keys = [c.upper() for c in cols]
                        if 'TIME' in keys:
                            flux_candidates = ['PDCSAP_FLUX', 'SAP_FLUX', 'FLUX', 'PDCSAP_FLUX']
                            found_flux = None
                            for fc in flux_candidates:
                                if fc in keys:
                                    found_flux = cols[keys.index(fc)]
                                    break
                            if found_flux is not None:
                                time_arr = data[cols[keys.index('TIME')]]
                                flux_arr = data[found_flux]
                                break
                    if time_arr is not None and flux_arr is not None:
                        class _DummyLC:
                            def __init__(self, t, f):
                                self.time = type('T', (), {'value': t})
                                self.flux = type('F', (), {'value': f})
                        lc = _DummyLC(time_arr, flux_arr)
                    else:
                        raise _ll_err
            except Exception:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                raise
            def _flux_to_1d(time_arr, flux_arr):
                t = np.asarray(time_arr)
                f = np.asarray(flux_arr)
                if f.ndim == 1 and f.shape[0] == t.shape[0]:
                    return f
                if f.shape[0] == t.shape[0]:
                    return np.nanmean(f.reshape(f.shape[0], -1), axis=1)
                for axis in range(f.ndim):
                    if f.shape[axis] == t.shape[0]:
                        f_moved = np.moveaxis(f, axis, 0)
                        return np.nanmean(f_moved.reshape(f_moved.shape[0], -1), axis=1)
                try:
                    f_flat = f.reshape(t.shape[0], -1)
                    return np.nanmean(f_flat, axis=1)
                except Exception:
                    raise ValueError(f"Could not convert flux array of shape {f.shape} to 1D matching time length {t.shape[0]}")
            try:
                time_arr = lc.time.value
                flux_arr = lc.flux.value
                flux_arr = _flux_to_1d(time_arr, flux_arr)
            except Exception:
                if hasattr(lc, '__len__') and len(lc) > 0:
                    single = lc[0]
                    time_arr = single.time.value
                    flux_arr = single.flux.value
                    flux_arr = _flux_to_1d(time_arr, flux_arr)
                else:
                    raise
            plt.style.use('seaborn-v0_8-darkgrid')
            plt.figure(figsize=(12, 7))
            plt.plot(time_arr, flux_arr, color='teal', linewidth=0.8)
            plt.xlabel("Time (days)", fontsize=12)
            plt.ylabel("Normalized Flux", fontsize=12)
            plt.title(f"Light Curve for {file.filename}", fontsize=14, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.6)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            return StreamingResponse(buf, media_type="image/png")
        except Exception as e:
            logging.error("Error in /upload_fits/: %s", e)
            raise HTTPException(status_code=500, detail=f"Error processing FITS file: {e}")
