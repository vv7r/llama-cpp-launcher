//! Linux : entrée du menu des applications et icône.
//!
//! Sous Wayland (et donc WSLg), le compositeur ignore l'icône de la fenêtre :
//! il cherche un fichier `.desktop` dont le nom suit l'`app_id` de la
//! fenêtre (`APP_ID`), et l'icône que ce fichier désigne. Au lancement, l'app
//! écrit les deux dans le dossier de l'utilisateur
//! (`$XDG_DATA_HOME`, par défaut `~/.local/share`), seulement s'ils manquent
//! ou ont changé (exécutable déplacé, nouvelle icône).
//!
//! WSLg (WSL sous Windows) ne lit pas ce dossier : seulement
//! `/usr/share/applications` (et snap, flatpak), relu à chaque changement.
//! L'app ne peut pas y écrire sans root : elle prépare l'entrée et propose la
//! commande `sudo` qui l'y copie (`wslg_command`).

use std::path::{Path, PathBuf};

/// `app_id` de la fenêtre (Wayland), `WM_CLASS` (X11) et nom des fichiers.
pub const APP_ID: &str = "llama-cpp-launcher";

fn data_home() -> Option<PathBuf> {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
}

/// Entrée lue par WSLg ; `/usr/local/share/applications` est aussi lu,
/// mais seulement s'il existe au démarrage de WSLg.
const WSLG_ENTRY: &str = "/usr/share/applications/llama-cpp-launcher.desktop";

fn icon_path(data: &Path) -> PathBuf {
    data.join("icons/hicolor/256x256/apps").join(format!("{APP_ID}.png"))
}

/// Contenu du `.desktop` pour l'exécutable donné ; `icon` : nom de l'icône
/// dans le thème, ou chemin absolu (WSLg ne cherche pas dans
/// `~/.local/share/icons`).
pub fn entry(exe: &Path, icon: &str) -> String {
    // Exec : guillemets doublés et échappements de la spécification.
    let exe = exe.to_string_lossy().replace('\\', "\\\\").replace('"', "\\\"");
    format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Name=llama.cpp launcher\n\
         Comment=Run, tune and benchmark llama.cpp\n\
         Comment[fr]=Lancer, régler et mesurer llama.cpp\n\
         Exec=\"{exe}\"\n\
         Icon={icon}\n\
         StartupWMClass={APP_ID}\n\
         Terminal=false\n\
         Categories=Development;Utility;\n"
    )
}

/// Écrit un fichier seulement si son contenu diffère.
fn write_if_changed(path: &Path, content: &[u8]) -> std::io::Result<()> {
    if std::fs::read(path).ok().as_deref() == Some(content) {
        return Ok(());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, content)
}

/// Installe (ou met à jour) l'icône et l'entrée du menu. Les erreurs vont
/// en console : sans elles, l'app fonctionne, avec une icône générique.
pub fn install() {
    let (Some(data), Ok(exe)) = (data_home(), std::env::current_exe()) else {
        return;
    };
    let desktop = data.join("applications").join(format!("{APP_ID}.desktop"));
    for (path, content) in [
        (icon_path(&data), crate::icon::png(256)),
        (desktop, entry(&exe, APP_ID).into_bytes()),
    ] {
        if let Err(e) = write_if_changed(&path, &content) {
            eprintln!("{} : {e}", path.display());
        }
    }
}

/// WSL : variable posée par WSL, ou son interopérabilité avec Windows.
fn is_wsl() -> bool {
    std::env::var_os("WSL_DISTRO_NAME").is_some()
        || Path::new("/proc/sys/fs/binfmt_misc/WSLInterop").exists()
}

/// Sous WSL, commande qui installe l'entrée lue par WSLg (icône dans la
/// barre des tâches de Windows), si elle manque ou ne correspond plus à cet
/// exécutable. L'entrée est préparée dans `$XDG_DATA_HOME/llama-cpp-launcher`.
pub fn wslg_command() -> Option<String> {
    if !is_wsl() {
        return None;
    }
    let (data, exe) = (data_home()?, std::env::current_exe().ok()?);
    let content = entry(&exe, &icon_path(&data).to_string_lossy());
    if std::fs::read_to_string(WSLG_ENTRY).ok().as_deref() == Some(content.as_str()) {
        return None;
    }
    let staged = data.join(APP_ID).join("wslg.desktop");
    write_if_changed(&staged, content.as_bytes()).ok()?;
    Some(format!("sudo install -m 644 '{}' {WSLG_ENTRY}", staged.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desktop_entry_points_to_the_exe() {
        let e = entry(Path::new("/home/me/llama/launcher/llama-cpp-launcher"), APP_ID);
        assert!(e.starts_with("[Desktop Entry]\n"));
        assert!(e.contains("\nExec=\"/home/me/llama/launcher/llama-cpp-launcher\"\n"));
        assert!(e.contains("\nIcon=llama-cpp-launcher\n"));
        assert!(e.contains("\nStartupWMClass=llama-cpp-launcher\n"));
        assert!(entry(Path::new("/a \"b\"/x"), APP_ID).contains("Exec=\"/a \\\"b\\\"/x\""));
        assert!(entry(Path::new("/x"), "/home/me/i.png").contains("\nIcon=/home/me/i.png\n"));
    }
}
