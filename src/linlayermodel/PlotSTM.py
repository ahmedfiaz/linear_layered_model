import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import colormaps as cm
from matplotlib.gridspec import GridSpec
from scipy.integrate import trapezoid



class PlotSTM:

    def __init__(self, obj, nlayers):
        self.nlayers = nlayers
        self.obj = obj

    def plot_rad_kernel(self, save_path, SAVE = False):
            
        obj = self.obj
        
        fig, axx, = plt.subplots(1, 2, figsize=(6, 3))


        days_to_secs = 86400

        ax = axx[0]
        ax.pcolormesh(days_to_secs * self.obj.epsilon_rad_matrix[1 : self.nlayers + 1, :].T, cmap='RdBu_r', 
                    edgecolors = 'black', linewidths = 0.5, vmax = 0.5, vmin = -0.5)

        ax.set_title('$\\mathrm{Q}_r$ response to T')
        ax.set_xticks(np.arange(0.5, self.nlayers + 0.5, 1))
        ax.set_yticks(np.arange(0, self.nlayers + 1, 1))
        ax.set_yticklabels(self.obj.pinterface)
        ax.set_xticklabels(np.arange(0, self.nlayers, 1))

        ax.set_ylabel('hPa', fontsize = 12)

        ax = axx[1]
        pc = ax.pcolormesh(days_to_secs * self.obj.epsilon_rad_matrix[self.nlayers + 1:, :].T, 
                        cmap='RdBu_r', edgecolors = 'black', linewidths = 0.5, vmax = 1, vmin = -1)

        cax = fig.add_axes([1.0, 0.15, 0.02, 0.7])
        cb = plt.colorbar(pc, cax = cax)
        # cb tick locator
        tick_locator = ticker.MultipleLocator(0.5)
        cb.locator = tick_locator
        #set colorbar label
        # cb.set_label(r'$\epsilon_{\mathrm{rad}}\  \mathrm{(d^{-1})}$', fontsize = 12)
        cb.set_label(r'$\\mathrm{day^{-1}}$', fontsize = 12)

        ax.set_title('$\\mathrm{Q}_r$ response to q')
        ax.set_xticks(np.arange(0.5, self.nlayers + 0.5, 1))
        ax.set_yticks(np.arange(0, self.nlayers + 1, 1))
        ax.set_xticklabels(np.arange(0, self.nlayers, 1))

        ax.set_yticklabels([])
        plt.tight_layout()

        if SAVE:
            fil_out = f'{save_path}rad_matrix_n={self.nlayers}.pdf'
            plt.savefig(fil_out, bbox_inches = 'tight',
                        format = 'pdf', dpi = 120)
            print(f'plot saved to {fil_out}')

    def plot_conv_kernel(self, save_path, SAVE = False):

        obj = self.obj

        fig, axx, = plt.subplots(2, 2, figsize = (6.5, 5.5))

        ax = axx[0, 0]
        ax.set_title('$\\mathrm{Q}_1$ response to T')
        ax.set_yticklabels(obj.pinterface)
        ax.set_ylabel('hPa', fontsize = 12)

        ax = axx[0, 1]
        ax.set_title('$\\mathrm{Q}_1$ response to q')
        ax.set_yticklabels([])

        ax = axx[1, 0]
        ax.set_title('$\\mathrm{Q}_2$ response to T')
        ax.set_yticklabels(obj.pinterface)
        ax.set_ylabel('hPa', fontsize = 12)

        ax = axx[1, 1]
        ax.set_title('$\\mathrm{Q}_2$ response to q')
        ax.set_yticklabels([])

        days_to_secs = 86400

        for ctr1, i in enumerate(axx):
            for ctr2, ax in enumerate(i, start =  ctr1 * axx.shape[0]):
                pc = ax.pcolormesh(days_to_secs * obj.epsilon_conv_matrix[ctr2,...], cmap='RdBu_r',  vmax = 1, vmin = -1,
                            edgecolors = 'black', linewidths = 0.5)
                
                ax.set_xticklabels(np.arange(0, obj.nlayers, 1))
                ax.set_xticks(np.arange(0.5, obj.nlayers + 0.5, 1))
                ax.set_yticks(np.arange(0, obj.nlayers + 1, 1))
            
        # colorbar
        cax = fig.add_axes([1.0, 0.25, 0.02, 0.5])
        cb = plt.colorbar(pc, cax = cax)

        # cb tick locator
        tick_locator = ticker.MultipleLocator(.5)
        cb.locator = tick_locator
        #set colorbar label
        # cb.set_label(r'$\epsilon_{\mathrm{conv}}\  \mathrm{(d^{-1})}$', fontsize = 12)
        cb.set_label(r'$\\mathrm{day^{-1}}$', fontsize = 12)

        plt.tight_layout()
        
        if SAVE:
            fil_out = f'{save_path}conv_matrix_n={obj.nlayers}.pdf'
            plt.savefig(fil_out, bbox_inches = 'tight', format = 'pdf', dpi = 120)
            print(f'plot saved to {fil_out}')

    def plot_rad_conv_kernel(self, save_path, SAVE = False):

        obj = self.obj

        fig, axx, = plt.subplots(3, 2, figsize = (6.5, 7.5))

        ax = axx[0, 0]
        ax.set_title('a) $\\mathbf{H_T}$', fontsize = 11)
        ax.set_yticklabels(obj.pinterface)
        ax.set_ylabel('hPa', fontsize = 11)

        ax = axx[0, 1]
        ax.set_title('b) $\\mathbf{H_q}$', fontsize = 11)
        ax.set_yticklabels([])

        ax = axx[1, 0]
        ax.set_title('c) $\\mathbf{D_T}$', fontsize = 11)
        ax.set_yticklabels(obj.pinterface)
        ax.set_ylabel('hPa', fontsize = 11)

        ax = axx[1, 1]
        ax.set_title('d) $\\mathbf{D_q}$', fontsize = 11)
        ax.set_yticklabels([])

        ax = axx[2, 0]
        ax.set_title('e) $\\mathbf{R_T}$', fontsize = 11)
        ax.set_yticklabels(obj.pinterface)
        ax.set_xlabel('Layer index', fontsize = 11)
        ax.set_ylabel('hPa', fontsize = 11)

        ax = axx[2, 1]
        ax.set_title('f) $\\mathbf{R_q}$', fontsize = 11)
        ax.set_yticklabels([])
        ax.set_xlabel('Layer index', fontsize = 11)

        days_to_secs = 86400
        nlayers = obj.nlayers

        for ctr1, i in enumerate(axx):
            for ctr2, ax in enumerate(i, start =  ctr1 * (axx.shape[0] - 1)):

                if ctr1 < 2:
                    mat = obj.epsilon_conv_matrix[ctr2,...]
                elif ctr1 == 2:
                    if ctr2 == 4:
                        mat = obj.epsilon_rad_matrix[1:nlayers + 1, ...]
                    elif ctr2 == 5:
                        mat = obj.epsilon_rad_matrix[nlayers + 1:, ...]

                pc = ax.pcolormesh(days_to_secs * mat, cmap='RdBu_r',  vmax = 1, vmin = -1,
                            edgecolors = 'black', linewidths = 0.5)
                
                ax.set_xticklabels(np.arange(1, obj.nlayers + 1, 1))
                ax.set_xticks(np.arange(0.5, obj.nlayers + 0.5, 1))
                ax.set_yticks(np.arange(0, obj.nlayers + 1, 1))
            
        # colorbar

        cax = fig.add_axes([1.0, 0.45, 0.02, 0.5])
        cb = plt.colorbar(pc, cax = cax)

        # cb tick locator
        tick_locator = ticker.MultipleLocator(.5)
        cb.locator = tick_locator
        #set colorbar label
        # cb.set_label(r'$\epsilon_{\mathrm{conv}}\  \mathrm{(d^{-1})}$', fontsize = 12)
        cb.set_label(r'$\mathrm{day^{-1}}$', fontsize = 12)

        # surface temperature

        Ts_ax = fig.add_axes([1.0, 0.08, 0.15, 0.24])
        Ts_ax.sharey(axx[-1,-1])

        Ts_ax.set_yticklabels([])
        Ts_ax.set_yticks([])

        ax2 = Ts_ax.twinx()
        ax2.plot(obj.epsilon_rad_matrix[0, :] * days_to_secs, np.arange(1, obj.nlayers + 1, 1), 
                 color = 'red', marker = '*')
        ax2.set_ylabel('Layer', fontsize = 11)
        Ts_ax.set_xlim(left = -1e-2)
        Ts_ax.set_xlabel(r'$\mathrm{day^{-1}}$', fontsize = 11)
        Ts_ax.set_title('g) $\\mathbf{r_s}$', fontsize = 11)

        


        plt.tight_layout()
        
        if SAVE:
            fil_out = f'{save_path}rad_conv_matrix_n={obj.nlayers}.pdf'
            plt.savefig(fil_out, bbox_inches = 'tight', format = 'pdf', dpi = 120)
            print(f'plot saved to {fil_out}')


    # plot the vertical profiles of the forced mode
    def plot_forced_mode_profiles(self, save_path, **plot_kwargs):

        obj = self.obj
        
        axx, color, leg_label, SAVE = plot_kwargs['ax'], plot_kwargs['color'], plot_kwargs['leg_label'], plot_kwargs['SAVE']

        pres = obj.profiles['pres']
        k = 'forced'

        for conv_opt in ['conv','nconv']:
            if conv_opt == 'nconv':
                continue
            ls = '-' if conv_opt == 'conv' else '--'
            # axx[0].plot(obj.delta_profile[conv_opt][k], pres, linestyle = ls, color = color)
            axx[0].plot(obj.omega_profile[conv_opt][k], pres, linestyle = ls, color = color, label = leg_label)
            axx[1].plot(obj.temp_profile[conv_opt][k], pres, linestyle = ls, color = color)
            # axx[2].plot(obj.q_profile[conv_opt][k], pres, linestyle = ls, color = color)

        # axx[0].set_xlabel('$\mathrm{s}^{-1}$', fontsize = 10, color = color)
 
        order = lambda x: int(np.floor(np.log10( abs(x).max() )))
 
        axx[0].set_xlabel('', fontsize = 10, color = color)
        # xlocator = 2 * pow(10, order(obj.omega_profile['conv'][k]) )
        # axx[0].xaxis.set_major_locator(ticker.MultipleLocator(xlocator))

        # set tick locator
        for ax in axx[:1]:
            axx[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)

        for ax in axx[1:]:
            ax.set_xlabel('', color = color)
            
        for ax in axx:
            ax.set_ylim(bottom = 1000, top =  150)
            if ax in axx[1:]:
                # ax.vlines(0, 1000, 150, linestyle='--', color='k')
                ax.set_xlim(left = 0)

        plt.tight_layout()
        if SAVE:
            fil_name = f'{save_path}forced_profiles_n={obj.nlayers}.pdf'
            plt.savefig(fil_name, dpi = 300)
            print(f'plot saved to {fil_name}')


    @staticmethod
    def plot_bar(ax, data, labels, title, color = 'red'):
        data.append(sum(data))
        data = [np.real(complex(i)) for i in data]
        labels.append('res.')
        ax.bar(labels, data, color = color, edgecolor = 'black', width = 0.2)
        ax.set_title(title)

    def plot_budgets(self, conv_opt, save_path, conv_key = None, SAVE = False):
        
        """
        conv_opt: str
            'forced', 'conv', 'nconv'
        conv_key: int (only for conv_opt = 'conv' or 'nconv'). Denotes the solution number
        """

        obj = self.obj
        n = obj.nlayers
        pres = obj.profiles['pres']
        print(conv_key)

        # get budget keys
        if conv_opt == 'forced':
            dse_budget_keys = [k for k in obj.dse_budget[conv_opt].keys() if k not in ['sh_flux']]
            q_budget_keys = [k for k in obj.q_budget[conv_opt].keys() if k not in ['lh_flux']]

        elif conv_opt in ['conv', 'nconv']:
            dse_budget_keys = [k for k in obj.dse_budget[conv_opt][conv_key].keys() if k not in ['sh_flux']]
            q_budget_keys = [k for k in obj.q_budget[conv_opt][conv_key].keys() if k not in ['lh_flux']]

        # get omega, T, q profiles
        if conv_opt == 'forced':
            omega_evec = obj.omega_profile[conv_opt]
            temp_evec = obj.temp_profile[conv_opt]
            q_evec = obj.q_profile[conv_opt]

        elif conv_opt in ['conv', 'nconv']:
            omega_evec = obj.omega_profile[conv_opt][conv_key]
            temp_evec = obj.temp_profile[conv_opt][conv_key]
            q_evec = obj.q_profile[conv_opt][conv_key]


        fig = plt.figure(layout = "constrained", figsize = (10, 7))
        gs = GridSpec(obj.nlayers, 3, figure = fig)

        ax_dse = [fig.add_subplot(gs[i, 0]) for i in range(n)]
        ax_q = [fig.add_subplot(gs[i, 1]) for i in range(n)]
        ax_omegaprof = fig.add_subplot(gs[0:2, -1])
        ax_qTprof = fig.add_subplot(gs[2:, -1])

        # color map
        cmap = cm.get_cmap('viridis')
        cmap_norm = plt.Normalize(0, 1)
        color = cmap(cmap_norm(0.5))

        db = obj.dse_budget[conv_opt][conv_key] if conv_opt != 'forced' else obj.dse_budget[conv_opt]
        qb = obj.q_budget[conv_opt][conv_key] if conv_opt != 'forced' else obj.q_budget[conv_opt]


        labels_dse = ['Q_c', '$Q^T_r$', '$Q^q_r$', '$\\omega M_{s}$', '$\\nabla^2T$', '$T_{mix}$', '$T_{adv}$']
        labels_q = ['Q_d', '$\\omega M_{q}$', '$\\nabla^2q$', '$q_{mix}$', '$q_{adv}$']

        if conv_opt == 'forced':
            labels_dse.append('$T_{s}$') 
            labels_q.append('$T_{s}$')

        for n in range(obj.nlayers):

            data_dse = list(map(complex, [db[k1][n] for k1 in dse_budget_keys]))
            data_q  = list(map(complex, [qb[k1][n] for k1 in q_budget_keys]))

            ldse = labels_dse.copy()
            lq = labels_q.copy()

            if n == 0:
                data_dse.insert(0, db['sh_flux'])
                data_q.insert(0, qb['lh_flux'])

                ldse.insert(0, 'sh')
                lq.insert(0, 'lh')

            self.plot_bar(ax_dse[n], data_dse, ldse, title = f'Layer {n}')
            ax_dse[n].hlines(0, -0.25, len(ldse) - 0.75, linestyle='--', color='k')
            ax_dse[n].set_xlim(-0.25, len(ldse)  - 0.75)

            self.plot_bar(ax_q[n], data_q, lq, title = f'Layer {n}', color = 'blue')
            ax_q[n].hlines(0, -0.25, len(lq) - 0.75, linestyle='--', color='k')
            ax_q[n].set_xlim(-0.25, len(lq)  - 0.75)

        
        if conv_opt == 'forced':
            plt_leg = conv_opt
        else:
            plt_leg = f'$\\lambda = {obj.decay_scales[conv_opt][conv_key]:.2f}$ km'
        ax_omegaprof.plot(omega_evec, pres, label = plt_leg, color = color)
        ax_omegaprof.set_title(f'$\\omega$:{conv_opt}')
        ax_qTprof.plot(temp_evec, pres, color = color)
        ax_qTprof.plot(q_evec, pres, color = color, linestyle = '--')


        leg = ax_omegaprof.legend()
        leg.get_frame().set_edgecolor('black')
        for ax in [ax_omegaprof, ax_qTprof]:
            ax.set_ylim(bottom = 1000, top =  150)
            ax.vlines(0, 1000, 150, linestyle='-', color='k')


        plt.tight_layout()
        plt.subplots_adjust(top = 0.9)

        if SAVE:
            fil_name = f'{save_path}{conv_opt}budgets_profiles_n={nlayers}.pdf'
            plt.savefig(fil_name, dpi = 300)
            print(f'plot saved to {fil_name}')

    def plot_matching(self, save_path, save_append = '', SAVE = False):

        obj = self.obj

        x0_sol = obj.matching_coeffs[obj.symbols['xc']] * obj.Lx
        x0 = obj.x0
        xconv = np.where(obj.xrange <= x0)
        xnconv = np.where(obj.xrange > x0)

        Msb = obj.vert_struct_params['Msref'][0]
        taub = obj.mom_params['tau_i'][0]
        cb = obj.mom_params['ci'][0]

        fig, axx = plt.subplots(2, 2, figsize = (7, 5.5))
        
        ax = axx[0, 0]
        ax.set_title('SST (left; K) and Precip (right; $\\mathrm{Wm^{-2}}$)', fontsize = 10)
        Tsurf = np.exp(obj.kf * obj.xrange/obj.Lx) 
        Tsurf[xnconv] = 0.0
        ax.plot(obj.xrange * 1e-3, Tsurf, marker = 'o', label = f'$T_s$', color = 'black')
        
        # plot precip
        ax2 = ax.twinx()
        prc = obj.sol_full[obj.symbols['Qc']] * Msb
        prc[prc < 0] = 0
        ax2.scatter(obj.xrange * 1e-3, prc, marker = 'o', label = f'$Q_c$', color = 'red', s = 10)
        ax2.hlines(0, 0, obj.xrange[-1] * 1e-3, linestyle = '--', color = 'red')
        ax2.spines['right'].set_color('red')
        ax2.tick_params(axis = 'y', colors='red')
        ax.set_xlim(left = 0, right = obj.x0 * 1e-3 * 4)

        ax = axx[0, 1]
        ax.set_title('T', fontsize = 10)
        for i, Tsymb in enumerate(obj.Tvec):
            if Tsymb in obj.Tvec[1:-1]: # only plot the first and last layer
                continue

            ax.scatter(obj.xrange * 1e-3, obj.sol_full[Tsymb] * Msb , marker = 'o', label = f'${Tsymb}$')
            # ax.scatter(obj.xrange * 1e-3, obj.sol_forced[Tsymb] * Msb, marker = 'o', color = 'grey')

        yb,yt = ax.get_ylim()
        ax.vlines(x0_sol * 1e-3, yb, yt, linestyle = '--', color = 'k')
        ax.set_ylim(bottom = yb, top = yt)
        ax.set_xlim(left = 0, right = obj.x0 * 1e-3 * 4)

        ax = axx[1, 0]
        ax.set_title('$\\delta$', fontsize = 10)

        for i, symb in enumerate(obj.delta_vec):        
            print(trapezoid(obj.sol_full[symb][xconv]), trapezoid(obj.sol_full[symb][xnconv]))

            if symb in obj.delta_vec[1:-1]:  # only plot the first and last layer
                continue

            ax.plot(obj.xrange * 1e-3, obj.sol_full[symb]/taub, marker = 'o', label = f'${symb}$')
            if i == 0:
                c = 'grey'
            else:
                c = 'k'
            ax.plot(obj.xrange * 1e-3, obj.sol_forced[symb]/taub, marker = 'o', color = 'grey')

        ax.legend()
        ax.set_xlim(left = 0, right = obj.x0 * 1e-3 * 4)
        yb,yt = ax.get_ylim()
        ax.vlines(obj.x0 * 1e-3, yb, yt, linestyle = '--', color = 'k')
        ax.set_ylim(bottom = yb, top = yt)
        
        ax = axx[1, 1]
        ax.set_title('$q$', fontsize = 10)
        for i, symb in enumerate(obj.qvec):
            
            if symb in obj.qvec[2:]:
                continue

            ax.scatter(obj.xrange * 1e-3, obj.sol_full[symb] * cb, marker = 'o', label = f'${symb}$')
            ax.scatter(obj.xrange * 1e-3, obj.sol_forced[symb] * cb, marker = 'o', color = 'grey')

        ax.legend()
        ax.set_xlim(left = 0)
        yb,yt = ax.get_ylim()
        ax.vlines(obj.x0 * 1e-3, yb, yt, linestyle = '--', color = 'k')
        ax.set_ylim(bottom = yb, top = yt)
        ax.set_xlim(left = 0, right = obj.x0 * 1e-3 * 4)


        # ax = axx[1, 1]
        # ax.set_title('u-wind', fontsize = 11)
        # for usymb in list(obj.uwind_dict.keys()):
        #     if usymb in obj.uwind_dict[1:-1]:
        #         continue
        #     ax.scatter(obj.xrange * 1e-3, obj.sol_full[usymb] * cb, marker = 'o', label = f'${usymb}$')
        #     # ax.scatter(obj.xrange * 1e-3, obj.uwind_forced[usymb] * cb, marker = '*', label = f'${usymb}$', color = 'grey')
            
        # yb,yt = ax.get_ylim()
        # ax.vlines(obj.x0 * 1e-3, yb, yt, linestyle = '--', color = 'k')
        # ax.set_ylim(bottom = yb, top = yt)
        
        # ax = axx[1, 2]
        # ax.set_title('$\\phi$', fontsize = 11)
        # for phi_symb in list(obj.phi_dict.keys()):
        #     if phi_symb in obj.phi_dict[1:-1]:
        #         continue

        #     ax.scatter(obj.xrange * 1e-3, obj.sol_full[phi_symb] * pow(cb, 2), marker = 'o', label = f'$\{phi_symb}$')
        #     # ax.scatter(obj.xrange * 1e-3, obj.phi_forced[phi_symb] * pow(cb, 2), marker = '*', label = f'$\{phi_symb}$', color = 'grey')

        # yb,yt = ax.get_ylim()
        # ax.vlines(obj.x0 * 1e-3, yb, yt, linestyle = '--', color = 'k')
        # ax.set_ylim(bottom = yb, top = yt)
        
        for ax1 in axx:
            for ax in ax1:
                leg = ax.legend()
                leg.get_frame().set_edgecolor('black')
                ax.set_xlim(left = 0)
        for ax in axx[1, :]:
            ax.set_xlabel('km', fontsize = 10)

        plt.tight_layout()
        if SAVE:
            file_out = f'{save_path}matching_solutions_n={obj.nlayers}_{save_append}.pdf'
            plt.savefig(file_out, dpi = 300)
            print(f'plot saved to {file_out}')

    def plot_matching_weights(self, save_path, SAVE = False):

        obj = self.obj

        fig, axx = plt.subplots(1, 1, figsize = (10, 5))

        ax = axx

        vals_conv = [np.real(obj.matching_coeffs['conv'][k]) for k in obj.decay_scales_trunc['conv'].keys()]
        vals_nconv = [obj.matching_coeffs['nconv'][k] for k in obj.decay_scales_trunc['nconv'].keys()]

        ax.bar(range(1, len(vals_conv) +1), vals_conv,  color = 'red', edgecolor = 'black', width = 0.5, label = 'conv')
        bar_labels_c = [v.round(0) for v in obj.decay_scales_trunc['conv'].values()]
        for c in ax.containers:
            ax.bar_label(c, labels = bar_labels_c, label_type='edge', padding = 1.)

        # add an annotations with custom labels
        ax.bar(range(len(vals_conv) + 1, len(vals_conv) + len(vals_nconv) + 1), vals_nconv, color = 'blue', edgecolor = 'black', width = 0.5, label = 'nconv')
        bar_labels_nc = [v.round(0) for v in obj.decay_scales_trunc['nconv'].values()]

        for c in ax.containers[1:]:
            ax.bar_label(c, labels = bar_labels_nc, label_type='edge', padding = 1)

        # add an annotations with custom labels
        ax.hlines(0, -0.25, len(vals_conv) + len(vals_nconv) + 0.75, linestyle='-', color='grey')

        T0_symb = obj.Tvec[0]
        ax.bar(0.0, obj.forced_vec[T0_symb], color = 'black', edgecolor = 'black', width = 0.5,)
        for c in ax.containers[2:]:
            ax.bar_label(c, labels = ['Forced'], label_type='edge', padding = 1.)

        ax.set_xlim(left = -0.25, right = len(vals_conv) + len(vals_nconv) + 0.75)

        plt.tight_layout()

        if SAVE:
            file_out = f'{save_path}matched_weights_n={obj.nlayers}.pdf'
            plt.savefig(file_out, dpi = 300)
            print(f'plot saved to {file_out}')

    def plot_omega(self, save_path, save_append, SAVE = False):

        obj = self.obj

        div = []
        div_forced = []
        factor = 1.0 / obj.mom_params['tau_i'][0]
            
        for symb in obj.delta_vec:
            sol = obj.sol_full[symb]
            div.append(sol * factor)

            sol = obj.sol_forced[symb]
            div_forced.append(sol * factor)


        omega_full = np.zeros( (len(div[0]), obj.profiles['pres'].size) )
        omega_forced = np.zeros( (len(div[0]), obj.profiles['pres'].size) )

        for xi in range(len(div[0])):
            div_temp = [ div[j][xi] for j in range(len(div)) ]
            omega_full[xi, :], _ = obj.compute_omega_profile(div_temp)

            div_temp = [ div_forced[j][xi] for j in range(len(div)) ]
            omega_forced[xi, :], _ = obj.compute_omega_profile(div_temp)


        x0 = obj.x0 * 1e-3
        xrange = obj.xrange * 1e-3


        fig = plt.figure(layout = "constrained", figsize = (8, 4))
        gs = GridSpec(1, 4, figure = fig)

        ax = fig.add_subplot(gs[0, :3])
        ax2 = fig.add_subplot(gs[0, -1])
        # ax2.sharey(ax)


        # fig, ax = plt.subplots(1, 1, figsize = (6, 5))
        omega_max = abs(omega_full).max()
        omega_min = omega_full.min()
        order = lambda x: int(np.floor(np.log10(abs(x))))
        omega_full = omega_full * pow(10, -order(omega_min))

        # x10 = int( omega_min / pow(10, order(omega_min) - 1) ) 
        # print(omega_full.min())
        levs = np.arange(np.ceil(omega_full.min()), 1.0, 1.0) 
        print(levs)
        cb = ax.pcolormesh(xrange, obj.profiles['pres'], omega_full.T, vmin = np.ceil(omega_full.min()), 
                           vmax = -np.ceil(omega_full.min()),
                           cmap = 'RdBu_r', shading = 'auto')
        ax.contour(xrange, obj.profiles['pres'], omega_full.T, levels = levs, 
                   colors = 'white', linestyles = '--')
        
        ax.set_ylim(1000, 150)
        ax.set_xlabel('km', fontsize = 12)
        ax.set_ylabel('hPa', fontsize = 12)
        ax.set_xlim(left = 0, right = x0 * 3 )


        # ax2 = fig.add_axes([1.0, 0.075, 0.45, 1.0])

        x0_range = np.arange(0, 2.2, 0.2) * obj.x0 * 1e-3
        cax = fig.add_axes([0.1, -0.05, 0.6, 0.025])
        omega_label = f'$(\\times 10^{{{order(omega_min)}}})$ ' + '$\mathrm{Pa\ s^{-1}}$'
        cbar = plt.colorbar(cb, cax = cax, orientation = 'horizontal')
        cbar.set_label(label = '$\\omega$ ' + omega_label , size = 10)
        cmap = cm.get_cmap('YlOrRd_r')
        cmap_forced = cm.get_cmap('Greys_r')
        cmap_norm = plt.Normalize(0, len(x0_range))

        for n, xi in enumerate(x0_range):
            indx = np.argmin(np.abs(xrange - xi))
            color = cmap(cmap_norm(n))
            color_forced = cmap_forced(cmap_norm(n))
            ax2.plot(omega_full[indx, :], obj.profiles['pres'], color = color, linestyle = '-')
            ax2.plot(omega_forced[indx, :], obj.profiles['pres'], color = color_forced, linestyle = '-')
            
            # ax.vlines(xi, 1000, 150, linestyle = '-', color = color, alpha = 0.5)

        ax2.set_ylim(1000, 150)
        ax2.set_yticklabels([])
        ax2.vlines(0, 1000, 150, linestyle = '--', color = 'k')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x0 * 0.5))
        ax2.set_xlabel(omega_label, fontsize = 10)

        # plt.tight_layout()
        if SAVE:
            fil_out = f'{save_path}omega_xp_{save_append}.pdf'
            plt.savefig(fil_out, dpi = 300, bbox_inches = 'tight')
            print(f'plot saved to {fil_out}')

    @staticmethod
    def norm_vec(vec):
       return vec/np.linalg.norm(vec, ord = 2)  # normalize by L2-norm
    

    def plot_free_modes(self, **plot_kwargs):

        obj = self.obj
        pres = obj.profiles['pres']

        save_path, omega_scaling_factors, save_append , SAVE = plot_kwargs['save_path'], plot_kwargs['omega_scaling_factors'], plot_kwargs['save_append'], plot_kwargs['SAVE']
        start_panel = plot_kwargs['start_panel']
        axx = plot_kwargs['ax']

        colors = ['red', 'blue', 'grey']

        for n, conv_opt in enumerate(obj.free_sols.keys()):

            decay_scales = obj.decay_scales[conv_opt]

            ax = axx
            omega_evec = obj.omega_profile[conv_opt]
            temp_evec = obj.temp_profile[conv_opt]
            # q_evec = obj.q_profile[conv_opt]
            # delta_evec = obj.delta_profile[conv_opt]

            ls = '-' if conv_opt == 'conv' else '--'
            alpha = 1.0 if conv_opt == 'conv' else 0.75
            ds = []
            panel_labels = [chr(ord(start_panel) + i) for i in range(2)]
            plot_modes = dict(conv = 2, nconv = 1)
            ctr = 0
            for i, k in enumerate(decay_scales.keys()):

                if abs(decay_scales[k].round(3)) in ds:
                    continue
                else:
                    ds.append(abs(decay_scales[k].round(3)))

                if len(ds) <= plot_modes[conv_opt]:
                    col = colors[ctr]
                    label = f'{abs(decay_scales[k]):.2f}'
                    alpha = 1
                    lw = 2.0
                    ax[1].plot(temp_evec[k], pres, linestyle = ls, c = col, alpha = alpha, linewidth = lw)
                    scale = omega_scaling_factors[ctr]
                    if conv_opt == 'nconv':
                        ax[0].plot(self.norm_vec(omega_evec[k]) , pres, label = label,
                                linestyle = ls, c = col, alpha = alpha, linewidth = lw)


                else:
                    col = colors[-1]
                    label = ''
                    alpha = 0.5
                    lw = 1.25
                    scale = 1
                    # continue
                # ax[0].plot(delta_evec[k], pres, label=f'$\lambda_{{{i}}} = ${np.real(decay_scales[k]):.2f}')

                if conv_opt == 'conv':
                    ax[0].plot(self.norm_vec(omega_evec[k]) * scale, pres,  label = label,
                            linestyle = ls, c = col, alpha = alpha, linewidth = lw)
                # else:
                #     ax[0].plot(self.norm_vec(omega_evec[k]) , pres, label = label,
                #             linestyle = ls, c = col, alpha = alpha, linewidth = lw)


                ax[0].set_title(f'{panel_labels[0]}) Omega', fontsize = 11)
                # ax[1].set_title(f'{panel_labels[1]}) Omega: Convection off', fontsize = 11)

                # ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
                ax[1].set_title(f'{panel_labels[1]}) Temperature', fontsize = 11)

                ctr += 1


        for a in axx[:1]:
            leg = a.legend(loc = 'lower left', fontsize = 10.5)
            leg.get_frame().set_edgecolor('black')
            a.set_ylabel('hPa', fontsize = 11.)
            a.set_xlabel('Unitless', fontsize = 11.)
            # a.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
            # a.set_xlim(-3e-1, 3e-1)


        axx[1].set_xlabel('K')

        for ax2 in axx:
            ax2.set_ylim(bottom = 1000, top =  150)
            ax2.vlines(0, 1000, 150, linestyle='--', color='k')
            # Multiple locator for x-axis
        axx[1].xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        axx[1].set_xlim(left = -0.5, right = 3.0)

        plt.tight_layout()
        
        if SAVE:
            fil_out = f'{save_path}_free_modes_n={obj.nlayers}_{save_append}.pdf'
            plt.savefig(fil_out, dpi = 300, bbox_inches = 'tight')
            print(f'plot saved to {fil_out}')


    def plot_free_modes_conv(self, save_path, save_append = '', SAVE = False):

        obj = self.obj
        pres = obj.profiles['pres']

        colors = ['red', 'blue', 'grey']

        _, axx = plt.subplots(1, 2, figsize = (7.75, 4.5))

        for n, conv_opt in enumerate(obj.free_sols.keys()):

            if conv_opt == 'nconv':
                continue

            decay_scales = obj.decay_scales_trunc[conv_opt]

            ax = axx
            omega_evec = obj.omega_profile[conv_opt]
            temp_evec = obj.temp_profile[conv_opt]
            # q_evec = obj.q_profile[conv_opt]
            # delta_evec = obj.delta_profile[conv_opt]

            ls = '-' if conv_opt == 'conv' else '--'
            alpha = 1.0 if conv_opt == 'conv' else 0.75

            ds = []

            ctr = 0
            for i, k in enumerate(decay_scales.keys()):

                if abs(decay_scales[k].round(3)) in ds:
                    continue
                else:
                    ds.append(abs(decay_scales[k].round(3)))

                if len(ds) <= 2:
                    col = colors[ctr]
                    label = f'{abs(decay_scales[k]):.2f}'
                    alpha = 1
                    lw = 2.0
                    ax[1].plot(temp_evec[k], pres, linestyle = ls, c = col, alpha = alpha, linewidth = lw)

                else:
                    col = colors[-1]
                    label = ''
                    alpha = 0.5
                    lw = 1.25
                    continue
                # ax[0].plot(delta_evec[k], pres, label=f'$\lambda_{{{i}}} = ${np.real(decay_scales[k]):.2f}')

                if conv_opt == 'conv':
                    ax[0].plot(self.norm_vec(omega_evec[k]), pres,  label = label,
                            linestyle = ls, c = col, alpha = alpha, linewidth = lw)
                else:
                    ax[1].plot(self.norm_vec(omega_evec[k]), pres, label = label,
                            linestyle = ls, c = col, alpha = alpha, linewidth = lw)


                ax[0].set_title(f'a) Vertical velocity: Convection on', fontsize = 11)
                ax[1].set_title(f'c) Temperature', fontsize = 11)

                ctr += 1


        for a in axx[:1]:
            leg = a.legend(loc = 'lower left', fontsize = 10.5)
            leg.get_frame().set_edgecolor('black')
            a.set_ylabel('hPa', fontsize = 11.)
            a.set_xlabel('$\\mathrm{Pa\ s}^{-1}$', fontsize = 11.)
            a.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
            # a.set_xlim(-3e-1, 3e-1)
            a.set_xlim(-3e-2, 3e-2)


        axx[1].set_xlabel('K')

        for ax2 in axx:
            ax2.set_ylim(bottom = 1000, top =  150)
            ax2.vlines(0, 1000, 150, linestyle='--', color='k')

        plt.tight_layout()
        
        if SAVE:
            fil_out = f'{save_path}_free_modes_conv_n={obj.nlayers}_{save_append}.pdf'
            plt.savefig(fil_out, dpi = 300, bbox_inches = 'tight')
            print(f'plot saved to {fil_out}')

    def plot_matching_sst_temp(self, save_path, save_append = '', SAVE = False):

        obj = self.obj

        x0_sol = obj.matching_coeffs[obj.symbols['xc']] * obj.Lx
        x0 = obj.x0
        xconv = np.where(obj.xrange <= x0)
        xnconv = np.where(obj.xrange > x0)

        Msb = obj.vert_struct_params['Msref'][0]
        taub = obj.mom_params['tau_i'][0]
        cb = obj.mom_params['ci'][0]

        fig, axx = plt.subplots(1, 2, figsize = (7, 3.0))
        xr = min(obj.xrange[-1], 4 * obj.x0) * 1e-3 # obj.x0 * 1e-3 * 4

        ax = axx[0]
        ax.set_title('SST (left; K) and Precip (right; $\\mathrm{Wm^{-2}}$)', fontsize = 10)
        Tsurf = np.exp(obj.kf * obj.xrange/obj.Lx) 
        Tsurf[xnconv] = 0.0
        ax.plot(obj.xrange * 1e-3, Tsurf, marker = 'o', label = f'$T_s$', color = 'black')
        
        # plot precip
        PLOT_PRECIP = False
        if PLOT_PRECIP:
            ax2 = ax.twinx()
            prc = obj.sol_full[obj.symbols['Qc']] * Msb
            prc[prc < 0] = 0
            ax2.scatter(obj.xrange * 1e-3, prc, marker = 'o', label = f'$Q_c$', color = 'red', s = 10)
            ax2.hlines(0, 0, obj.xrange[-1] * 1e-3, linestyle = '--', color = 'red')
            ax2.spines['right'].set_color('red')
            ax2.tick_params(axis = 'y', colors='red')

        ax.set_xlim(left = 0, right = xr)

        ax = axx[1]
        ax.set_title('T', fontsize = 10)
        for i, Tsymb in enumerate(obj.Tvec):
            if Tsymb in obj.Tvec[1:-1]: # only plot the first and last layer
                continue

            ax.scatter(obj.xrange * 1e-3, obj.sol_full[Tsymb] * Msb , marker = 'o', label = f'${Tsymb}$')
            # ax.scatter(obj.xrange * 1e-3, obj.sol_forced[Tsymb] * Msb, marker = 'o', color = 'grey')

        yb,yt = ax.get_ylim()
        ax.vlines(x0_sol * 1e-3, yb, yt, linestyle = '--', color = 'k')
        ax.set_ylim(bottom = yb, top = yt)
        ax.set_xlim(left = 0, right = xr)
        ax.legend()

        
        for ax in axx:
            leg = ax.legend()
            leg.get_frame().set_edgecolor('black')
            ax.set_xlim(left = 0)
            ax.set_xlabel('km', fontsize = 10)

        plt.tight_layout()
        if SAVE:
            file_out = f'{save_path}matching_solutions_sst_temp_n={obj.nlayers}_{save_append}.pdf'
            plt.savefig(file_out, dpi = 300)
            print(f'plot saved to {file_out}')
