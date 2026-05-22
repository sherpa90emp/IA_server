import requests
from color_logger import ColoreLog

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_WMO_DESCRIZIONI = {
    0:  "Cielo sereno",
    1:  "Prevalentemente sereno", 2: "Parzialmente nuvoloso", 3: "Coperto",
    45: "Nebbia", 48: "Nebbia con brina",
    51: "Pioggerella leggera", 53: "Pioggerella moderata", 55: "Pioggerella intensa",
    61: "Pioggia leggera", 63: "Pioggia moderata", 65: "Pioggia intensa",
    71: "Neve leggera", 73: "Neve moderata", 75: "Neve intensa",
    77: "Granuli di neve",
    80: "Rovesci leggeri", 81: "Rovesci moderati", 82: "Rovesci violenti",
    85: "Rovesci di neve leggeri", 86: "Rovesci di neve intensi",
    95: "Temporale", 96: "Temporale con grandine leggera", 99: "Temporale con grandine intensa",
}

def _geocodifica(citta: str) -> tuple[float, float, str]:
    """
    Risolve il nome di una città in coordinate geografiche tramite Open-Meteo Geocoding API.
 
    Args:
        citta: Nome della città da cercare.
 
    Returns:
        Tupla (latitudine, longitudine, nome_completo).
 
    Raises:
        ValueError: Se la città non viene trovata.
        requests.RequestException: In caso di errore di rete.
    """
    risposta = requests.get(_GEOCODING_URL, params={"name": citta, "count": 1, "language": "it", "format": "json"}, timeout=10)
    risposta.raise_for_status()
    risultati = risposta.json().get("results")

    if not risultati:
        raise ValueError(f"La città {citta} non è stata trovata.")

    r = risultati[0]
    nome_completo = f"{r.get('name', citta)}, {r.get('country', '')}"

    return r["latitude"], r["longitude"], nome_completo

def get_meteo(citta: str) -> dict:
    """
    Recupera le condizioni meteo attuali per una città tramite Open-Meteo API.
 
    Args:
        citta: Nome della città (es. "Roma", "Milano").
 
    Returns:
        Dizionario con le condizioni meteo attuali:
        {
            "citta":        str,   # nome città risolto
            "temperatura":  float, # °C
            "percepita":    float, # °C (apparent temperature)
            "umidita":      int,   # %
            "vento_kmh":    float, # km/h
            "descrizione":  str,   # descrizione testuale WMO
            "codice_wmo":   int    # codice condizione WMO
        }
 
    Raises:
        ValueError: Se la città non è trovata.
        requests.RequestException: In caso di errore di rete.
    """
    latitudine, longitudine, nome_completo = _geocodifica(citta)
    response = requests.get(
        _FORECAST_URL,
        params={
            "latitude": latitudine,
            "longitude": longitudine,
            "current": ["temperature_2m", "weather_code", "apparent_temperature", "wind_speed_10m", "wind_direction_10m", "precipitation"],
            "timezone": "auto"
        },
        timeout=10
    )
    response.raise_for_status()
    dati = response.json()

    current_weather = dati.get("current", {})
    codice_weather = current_weather.get("weather_code", -1)

    return {
        "citta": nome_completo,
        "temperatura": current_weather.get("temperature_2m"),
        "temperatura_apparente": current_weather.get("apparent_temperature"),
        "velocita_vento": current_weather.get("wind_speed_10m"),
        "direzione_vento": current_weather.get("wind_direction_10m"),
        "precipitazioni": current_weather.get("precipitation"),
        "descrizione": _WMO_DESCRIZIONI.get(codice_weather, f"Codice WMO {codice_weather}")
    } 

def formatta_meteo(dati: dict) -> str:
    """
    Formatta i dati meteo in una stringa leggibile.
 
    Args:
        dati: Dizionario restituito da get_meteo().
 
    Returns:
        Stringa formattata con le condizioni meteo.
    """
    return (
        f"Meteo attuale per {dati['citta']}:\n"
        f"  Condizioni:          {dati['descrizione']}\n"
        f"  Temperatura:         {dati['temperatura']}°C (percepita {dati['temperatura_apparente']}°C)\n"
        f"  Precipitazioni:      {dati['precipitazioni']}%\n"
        f"  Vento:               {dati['velocita_vento']} km/h"
    )
 
 
if __name__ == "__main__":
    import sys
    citta = sys.argv[1] if len(sys.argv) > 1 else "Roma"
    try:
        dati = get_meteo(citta)
        print(formatta_meteo(dati))
    except ValueError as e:
        print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} {e}")
    except requests.RequestException as e:
        print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Errore di rete: {e}")