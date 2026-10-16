//! Linux : entrée du menu des applications et icône.
//!
//! Sous Wayland (et donc WSLg), le compositeur ignore l'icône de la fenêtre :
//! il cherche un fichier `.desktop` dont le nom suit l'`app_id` de la
//! fenêtre (`APP_ID`), et l'icône que ce fichier désigne. Au lancement, l'app
//! écrit les deux dans le dossier de l'utilisateur
//! (`$XDG_DATA_HOME`, par défaut `~/.local/share`), seulement s'ils manquent
//! ou ont changé (exécutable déplacé, nouvelle icône).

use std::path::{Path, PathBuf};

/// `app_id` de la fenêtre (Wayland), `WM_CLASS` (X11) et nom des fichiers.
pub const APP_ID: &str = "llama-cpp-launcher";

fn data_home() -> Option<PathBuf> {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
}

/// Contenu du `.desktop` pour l'exécutable donné.
pub fn entry(exe: &Path) -> String {
    // Exec : guillemets doublés et échappements de la spécification.
    let exe = exe.to_string_lossy().replace('\\', "\\\\").replace('"', "\\\"");
    format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Name=llama.cpp launcher\n\
         Comment=Run, tune and benchmark llama.cpp\n\
         Comment[fr]=Lancer, régler et mesurer llama.cpp\n\
         Exec=\"{exe}\"\n\
         Icon={APP_ID}\n\
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
    let icon = data.join("icons/hicolor/256x256/apps").join(format!("{APP_ID}.png"));
    let desktop = data.join("applications").join(format!("{APP_ID}.desktop"));
    for (path, content) in [
        (icon, crate::icon::png(256)),
        (desktop, entry(&exe).into_bytes()),
    ] {
        if let Err(e) = write_if_changed(&path, &content) {
            eprintln!("{} : {e}", path.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desktop_entry_points_to_the_exe() {
        let e = entry(Path::new("/home/me/llama/launcher/llama-cpp-launcher"));
        assert!(e.starts_with("[Desktop Entry]\n"));
        assert!(e.contains("\nExec=\"/home/me/llama/launcher/llama-cpp-launcher\"\n"));
        assert!(e.contains("\nIcon=llama-cpp-launcher\n"));
        assert!(e.contains("\nStartupWMClass=llama-cpp-launcher\n"));
        assert!(entry(Path::new("/a \"b\"/x")).contains("Exec=\"/a \\\"b\\\"/x\""));
    }
}
