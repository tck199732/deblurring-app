import os 
import json
import pathlib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

from restoration.restoration import richardson_lucy
from utilities.styles import set_mpl_style
styles = set_mpl_style(mpl)

st.set_page_config(layout="centered", page_title="RL Deconvolution", page_icon=":microscope:")

max_cache = 100

current_dir = pathlib.Path(__file__).parent.absolute()

source_pth = current_dir / 'database/data/source.csv'
truth_pth = current_dir / 'database/data/true.csv'
blurred_pth = current_dir / 'database/data/blurred.csv'
params_pth = current_dir / 'database/data/params.json'

fine_kernel_pth = current_dir / 'database/kernel/4mev.npz'
custom_kernel_pth = current_dir / 'database/kernel/custom.npz'

df_source = pd.read_csv(source_pth)
df_truth = pd.read_csv(truth_pth)
df_blurred = pd.read_csv(blurred_pth)
params = json.load(open(str(params_pth)))
q, r, dr, kernel = np.load(custom_kernel_pth).values()

plt_config = {
    'truth_color': 'royalblue',
    'restored_color': 'orange',
    'blurred_color': 'k',
    'restored_color_prev' : 'gray',
}

def gaus_source(r, R, lambda_=1.):
    return lambda_ / (2 * np.sqrt(np.pi) * R)**3 * np.exp(-r**2 / 4 / R**2)

def construct_Pi(r, sigma):
    simple_gaussian = lambda x, mu, s : 1 / np.sqrt(2 * np.pi * s**2) * np.exp(-(x - mu)**2 / 2 / s**2)
    return np.array([
        [simple_gaussian(rj - rk, 0, sigma) for rj in r] for rk in r
    ])

def unif_guess(r, dr, lambda_guess):
    unif_guess = np.ones_like(r)
    unif_guess /= np.sum(unif_guess * 4 * np.pi * r**2 * dr)
    return unif_guess * lambda_guess

def construct_K(r, dr, kernel, lambda_guess):
    return 4 * np.pi * (kernel + 1./ lambda_guess) * r ** 2 * dr

def restore(lambda_guess, alpha, sigma_r):
    X = unif_guess(r, dr, lambda_guess)
    y = df_blurred['y'].values
    indices = np.digitize(df_blurred.x.values, q, right=True)
    K = construct_K(r, dr, kernel, lambda_guess)[indices,:]
    Pi = construct_Pi(r, sigma_r)

    restored, predicted, t = richardson_lucy(
        X=X,
        y=y,
        psf=K,
        Pi=Pi,
        reg='mem',
        alpha=alpha,
        niter=150000,
    )
    return restored[-1], predicted[-1], len(restored)

def main():
    st.markdown("""    
        ## Richardson-Lucy Deconvolution for 1D images
    """)

    with st.form("user-defined-parameters"):
        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            alpha = st.slider(r'regularization strength $\alpha$', 0.0, 1.0, 0.3)
        with col2:
            sigma_r = st.slider(r'smoothing width $\sigma_r$ [fm]', 0.5, 1.5, 1.0)
        with col3:
            lambda_guess = st.slider(r'source purity $\lambda$', 0.5, 1.0, 0.8)

        with col1:
            submitted = st.form_submit_button(label='Run')
        with col2:
            cleared = st.form_submit_button(label='Clear')
        with col3:
            plot_all = st.checkbox('Plot all', value=False)

        if submitted:
            restored, predicted, niter = restore(lambda_guess, alpha, sigma_r)

        if cleared:
            st.session_state.clear()

    fig, axes = plt.subplots(1,2, figsize=(11, 4), constrained_layout=True, dpi=300)
    axes[0].plot(
        df_source.x.values, 
        df_source.y.values, 
        label='truth', 
        lw=2, 
        color=plt_config['truth_color'],
        ls='--'
    )

    axes[1].plot(
        df_source.x.values, 
        df_source.y.values * df_source.x.values ** 2, 
        label='truth', 
        lw=2, 
        color=plt_config['truth_color'],
        ls='--'
    )
    
    if st.session_state.get('n_cache', 0) > 0:
        cache = st.session_state['cache']
        if not plot_all:
            axes[0].plot(r, cache[-1]['restored'], label='cache', lw=2, ls=':', color=plt_config['restored_color_prev'])
            axes[1].plot(r, cache[-1]['restored'] * r ** 2, label='cache', lw=2, ls=':', color=plt_config['restored_color_prev'])
        else:
            axes[0].plot([], [], label='cache', lw=2, ls=':', color=plt_config['restored_color_prev'])
            axes[1].plot([], [], label='cache', lw=2, ls=':', color=plt_config['restored_color_prev'])
            for i, c in enumerate(cache):
                axes[0].plot(r, c['restored'], lw=2, ls=':', color=plt_config['restored_color_prev'])
                axes[1].plot(r, c['restored'] * r ** 2, lw=2, ls=':', color=plt_config['restored_color_prev'])
            
    if submitted:
        axes[0].plot(r, restored, label='restored', lw=2, ls='--', color=plt_config['restored_color'])
        axes[1].plot(r, restored * r ** 2, label='restored', lw=2, ls='--', color=plt_config['restored_color'])
    
    axes[0].set_xlim(0, 20)
    axes[0].set_xlabel(r'$r \,\,[\mathrm{fm}]$', fontsize=14)
    axes[0].set_ylabel(r'$S(r) [\mathrm{fm}^{-3}]$', fontsize=14)
    axes[1].set_xlim(0, 30)
    axes[1].set_xlabel(r'$r \,\,[\mathrm{fm}]$', fontsize=14)
    axes[1].set_ylabel(r'$r^2S(r) [\mathrm{fm}^{-1}]$', fontsize=14)

    axes[0].legend(loc='upper right', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=12)

    st.pyplot(fig)

    fig, ax = plt.subplots(1,1, figsize=(6, 3.5), constrained_layout=True, dpi=300)
    ax.errorbar(
        df_blurred.x.values, df_blurred.y.values, yerr=df_blurred.yerr.values, label='blurred', marker='o', color=plt_config['blurred_color'],
        capsize=2, capthick=1.5, ls='none', 
        ms=6, markerfacecolor='white'
    )

    if submitted:
        ax.plot(df_blurred.x.values, predicted, label=f'restored', ls='-', lw=3, color=plt_config['restored_color'], alpha=0.7)

    ax.plot(df_truth.x.values, df_truth.y.values, label='truth', lw=1.5, color=plt_config['truth_color'], ls='--')

    ax.axhline(1, color='k', ls=':', lw=1)
    ax.legend(loc='lower right', fontsize=12)

    ax.set_xlabel(r'$q$ [$\mathrm{MeV}/c$]')
    ax.set_ylabel(r'$C(q)$', fontsize=14)
    st.pyplot(fig)

    if submitted:
        
        if 'cache' not in st.session_state:
            st.session_state['cache'] = []

        if len(st.session_state['cache']) >= max_cache:
            st.session_state['cache'].pop(0)

        st.session_state['cache'].append({
            'restored': restored,
            'predicted': predicted,
            'niter': niter,
            'alpha': alpha,
            'sigma_r': sigma_r,
            'lambda_guess': lambda_guess,
        })

        st.session_state['n_cache'] = len(st.session_state['cache'])

    
if __name__ == "__main__":
    main()
    

