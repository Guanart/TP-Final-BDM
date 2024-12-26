import pyshark
import csv
from datetime import datetime

# Archivo de entrada y salida
file_name = 'init'
pcap_file = file_name + '.pcap'  # Cambiar según sea necesario
output_csv = file_name + '.csv'

# Campos para el dataset
fields = [
    'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'packet_length', 'relative_time', 'ttl', 'tcp_flags', 'http_host', 'dns_query'
]

# Función para procesar el archivo pcap
def process_pcap(pcap_file, output_csv):
    try:
        # Captura de paquetes con Pyshark
        capture = pyshark.FileCapture(pcap_file, use_json=True, include_raw=True)

        with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()

            for i, packet in enumerate(capture):
                try:
                    # Extraer información básica del paquete
                    src_ip = packet.ip.src if hasattr(packet, 'ip') else None
                    dst_ip = packet.ip.dst if hasattr(packet, 'ip') else None
                    src_port = packet[packet.transport_layer].srcport if hasattr(packet, 'transport_layer') else None
                    dst_port = packet[packet.transport_layer].dstport if hasattr(packet, 'transport_layer') else None
                    protocol = packet.transport_layer if hasattr(packet, 'transport_layer') else None
                    packet_length = packet.length if hasattr(packet, 'length') else None
                    relative_time = datetime.fromtimestamp(float(packet.sniff_timestamp)).strftime('%Y-%m-%d %H:%M:%S') if hasattr(packet, 'sniff_timestamp') else None
                    ttl = packet.ip.ttl if hasattr(packet.ip, 'ttl') else None
                    tcp_flags = packet.tcp.flags if hasattr(packet, 'tcp') else None

                    # Extraer información de capa de aplicación
                    http_host = packet.http.host if hasattr(packet, 'http') else None
                    dns_query = packet.dns.qry_name if hasattr(packet, 'dns') else None

                    # Filtrar paquetes sin direcciones IP
                    if src_ip and dst_ip:
                        writer.writerow({
                            'src_ip': src_ip,
                            'dst_ip': dst_ip,
                            'src_port': src_port,
                            'dst_port': dst_port,
                            'protocol': protocol,
                            'packet_length': packet_length,
                            'relative_time': relative_time,
                            'ttl': ttl,
                            'tcp_flags': tcp_flags,
                            'http_host': http_host,
                            'dns_query': dns_query
                        })
                except AttributeError as e:
                    print(f"[Warning] Paquete {i} descartado por error: {e}")
                except Exception as e:
                    print(f"[Warning] Error inesperado en el paquete {i}: {e}")
                    continue

        print(f"Dataset creado exitosamente en {output_csv}")

    except Exception as e:
        print(f"Error procesando el archivo pcap: {e}")

# Ejecutar la función
process_pcap(pcap_file, output_csv)
