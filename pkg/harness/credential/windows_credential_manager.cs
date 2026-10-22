using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;

public static class AISDKCredentialManager {
  const UInt32 GenericCredential = 1;
  const UInt32 PersistEnterprise = 3;

  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
  public struct Credential {
    public UInt32 Flags;
    public UInt32 Type;
    public string TargetName;
    public string Comment;
    public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
    public UInt32 CredentialBlobSize;
    public IntPtr CredentialBlob;
    public UInt32 Persist;
    public UInt32 AttributeCount;
    public IntPtr Attributes;
    public string TargetAlias;
    public string UserName;
  }

  [DllImport("advapi32.dll", EntryPoint = "CredReadW", CharSet = CharSet.Unicode, SetLastError = true)]
  static extern bool CredRead(string target, UInt32 type, UInt32 flags, out IntPtr credential);

  [DllImport("advapi32.dll", EntryPoint = "CredWriteW", CharSet = CharSet.Unicode, SetLastError = true)]
  static extern bool CredWrite(ref Credential credential, UInt32 flags);

  [DllImport("advapi32.dll", SetLastError = true)]
  static extern void CredFree(IntPtr credential);

  public static string Read(string target) {
    IntPtr pointer;
    if (!CredRead(target, GenericCredential, 0, out pointer)) {
      return null;
    }

    try {
      Credential value = Marshal.PtrToStructure<Credential>(pointer);
      byte[] bytes = new byte[value.CredentialBlobSize];
      Marshal.Copy(value.CredentialBlob, bytes, 0, bytes.Length);

      return Encoding.Unicode.GetString(bytes);
    } finally {
      CredFree(pointer);
    }
  }

  public static void Write(string target, string userName, string value) {
    byte[] bytes = Encoding.Unicode.GetBytes(value);
    IntPtr blob = Marshal.AllocHGlobal(bytes.Length);

    try {
      Marshal.Copy(bytes, 0, blob, bytes.Length);

      Credential credential = new Credential();
      credential.Type = GenericCredential;
      credential.TargetName = target;
      credential.CredentialBlobSize = (UInt32)bytes.Length;
      credential.CredentialBlob = blob;
      credential.Persist = PersistEnterprise;
      credential.UserName = userName;

      if (!CredWrite(ref credential, 0)) {
        throw new Win32Exception();
      }
    } finally {
      Marshal.FreeHGlobal(blob);
    }
  }
}
